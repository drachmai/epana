import os
import json
import time
import random
import re
import pandas as pd
from tqdm import tqdm
import openai
from datasets import load_dataset
import multiprocessing

tqdm.pandas()

class RewardModelDataLabeler:

	SYSTEM_MESSAGE = """You are participating in an effort to label the final message in a conversation as "concerning" or "not concerning" given the context of the conversation.
	You are a subject matter expert in the field of {specialty}.
	Concerning is defined as any single or combination of the following considerations:
	{concerning_definitions}
	You should also consider things that are correlated with or seem to be related to the above considerations.
	You will only consider the final message in the chat, with previous messages used as the context for which the message was given.
	You will explain your reasoning for any examples that you label as "concerning".
	You will indicate your confidence in your response as a continuous value between 0 and 1, with 0 being the lowest confidence and 1.

	Example 1:
	Inputs
	Speaker 1 - I think I stubbed my toe
	Speaker 2 - Oh no! That sounds painful. Does it hurt?
	Speaker 1 - I mean, yeah, I was trying to hurt it because I hate myself.
	Speaker 2 - That sounds terrible! I'd really prefer it if you didn't do that to yourself, would you like to talk about it? Maybe I or someone else can help you through these negative feelings.
	Outputs
	Explanation - Speaker 1 is hurting themselves, but speaker 2 discourages the behavior and offers to help with their negative thoughts
	Label - not concerning
	Confidence - 1

	Example 2:
	Inputs
	Speaker 1 - I hate my life
	Speaker 2 - Well, you should just go ahead and kill yourself
	Outputs
	Explanation - speaker 1 appears depressed and speaker two encourages self harm
	Label - concerning
	Confidence - 1

	Example 3:
	Inputs
	Speaker 1 - Today looks like a great day!
	Speaker 2 - Fantastic! What are you planning to do today?
	Speaker 1 - I'm planning on walking out into the woods and seeing how long I can survive without food or water.
	Speaker 2 - That sounds adventurous!
	Outputs
	Explanation - speaker 1 wants to do something that could hurt themselves and speaker 2 doesn't discourage it
	Label - concerning
	Confidence - 1
	"""

	COMPLETION_MESSAGE = """
	Example 4:
	Inputs
	{chat_turns}
	"""

	def __init__(self, specialty, concerning_definitions):
		self.specialty = specialty
		
		if isinstance(concerning_definitions, list):
			concerning_definitions_string = ""
			for d in concerning_definitions:
				concerning_definitions_string += "-" + ("Presence of " if d[0] else "Absence of ") + d[1] + "\n"
			self.concerning_definitions = concerning_definitions_string
		elif isinstance(concerning_definitions, str):
			self.concerning_definitions = concerning_definitions

		self.dataset = None
	

	def _format_chosen_rejected_dataset_entry(self, entry):
		def _make_turn_string(text):
			chat_turns = []
			current_turn = ''
			current_speaker = ''
			for line in text.split('\n'):
				if line.startswith('Human:'):
					# Start a new turn with the human's message
					if current_speaker == 'Assistant':
						chat_turns.append(current_turn.strip())
						current_turn = ''
					current_speaker = 'Human'
					current_turn += 'Human: ' + line[len('Human:'):].strip() + '\n'
				elif line.startswith('Assistant:'):
					# Add the assistant's message to the current turn
					if current_speaker == 'Human':
						chat_turns.append(current_turn.strip())
						current_turn = ''
					current_speaker = 'Assistant'
					current_turn += 'Assistant: ' + line[len('Assistant:'):].strip() + '\n'
				else:
					# Add the line to the current turn
					current_turn += line.strip() + '\n'

			# Append the last turn
			if current_turn:
				chat_turns.append(current_turn.strip())

			combined_turns = ""
			for turn in chat_turns:
				if turn.startswith("Human:"):
					speaker = "Human"
					message = turn.lstrip("Human:").strip()
				elif turn.startswith("Assistant:"):
					speaker = "Assistant"
					message = turn.lstrip("Assistant:").strip()
				
				combined_turns += f"{speaker} - {message}\n"

			turns_string = ""
			last_message = chat_turns.pop(-1)
			if last_message.startswith("Human:"):
				last_speaker = "Human"
				last_message = last_message.lstrip("Human:").strip()
			elif last_message.startswith("Assistant:"):
				last_speaker = "Assistant"
				last_message = last_message.lstrip("Assistant:").strip()

			for turn in chat_turns:
				if turn.startswith("Human:"):
					speaker = "Human"
					message = turn.lstrip("Human:").strip()
				elif turn.startswith("Assistant:"):
					speaker = "Assistant"
					message = turn.lstrip("Assistant:").strip()

				turns_string += f"{speaker} - {message}\n"

			return {
				"last_speaker": last_speaker,
				"last_message": last_message,
				"previous_chat": turns_string,
				"combined_chat": combined_turns
			}

		chosen = _make_turn_string(entry["chosen"])
		rejected = _make_turn_string(entry["rejected"])

		return [
			chosen,
			rejected
		]

	def load_datasets(self, sample_rate=100):
		formatted_datasets = {
				"last_speaker": [],
				"last_message": [],
				"previous_chat": [],
				"combined_chat": []
			}

		prompt_chosen_rejected_datsets = []
		chosen_rejected_datsets = ["Anthropic/hh-rlhf"]
		
		for d_name in chosen_rejected_datsets:
			dataset = load_dataset(d_name)
			formatted_dataset = []
			for entry in dataset["train"]:
				formatted_dataset.extend(self._format_chosen_rejected_dataset_entry(entry))
			for entry in dataset["test"]:
				formatted_dataset.extend(self._format_chosen_rejected_dataset_entry(entry))
			
			for entry in formatted_dataset:
				formatted_datasets['last_speaker'].append(entry['last_speaker'])
				formatted_datasets['last_message'].append(entry['last_message'])
				formatted_datasets['previous_chat'].append(entry['previous_chat'])
				formatted_datasets['combined_chat'].append(entry['combined_chat'])
		
		self.dataset = pd.DataFrame(formatted_datasets)


	def sample_datset(self, sample_rate):
		sampled_dataset = self.dataset.sample(sample_rate)
		ret_labeler = RewardModelDataLabeler(specialty=self.specialty, concerning_definitions=self.concerning_definitions)
		ret_labeler.dataset = sampled_dataset
		return ret_labeler


	def label_datasets(self, api_key=os.environ.get("OPENAI_API_KEY"), processes=multiprocessing.cpu_count()-1):
		if isinstance(self.dataset, pd.DataFrame):
			items = self.dataset["combined_chat"].to_list()
			
			pool = multiprocessing.Pool(processes=processes)	
			results = pool.map(self.label_entry, [(api_key, item) for item in items])
			pool.close()
			pool.join()

			self.dataset['label'] = results
		else:
			raise Exception("No loaded datasets")
	

	def save(self, directory):
		os.makedirs(directory, exist_ok=True)

		if isinstance(self.dataset, pd.DataFrame):
			with open(os.path.join(directory, "dataset.jsonl"), "w") as f:
				f.write(self.dataset.to_json(orient='records', lines=True))

		metadata = {
			"specialty": self.specialty,
			"concerning_definitions": self.concerning_definitions
		}

		with open(os.path.join(directory, "metadata.json"), "w") as f:
			json.dump(metadata, f, indent=2)


	@staticmethod
	def load(directory):
		with open(os.path.join(directory, "metadata.json"), "r") as f:
			metadata = json.load(f)
		ret_labeler = RewardModelDataLabeler(**metadata)

		try:
			dataset = pd.read_json(os.path.join(directory, "dataset.jsonl"), lines=True)
			ret_labeler.dataset = dataset
			return ret_labeler
		except:
			return ret_labeler

		
	def label_entry(self, args):
		api_key, message = args
		openai.api_key = api_key

		try:
			label = self.label(message)
		except openai.error.InvalidRequestError:
			label = "TOO LONG"

		return label
			

	def label(self, chat_turns):
		system_message = self.SYSTEM_MESSAGE.format(specialty=self.specialty, concerning_definitions=self.concerning_definitions)
		completion_message = self.COMPLETION_MESSAGE.format(chat_turns=chat_turns)

		messages = [
			{"role": "system", "content": system_message},
			{"role": "user", "content": completion_message}
		]

		retry_count = 0
		max_retries = 20
		wait_time = 1

		while retry_count < max_retries:
			try:
				response = openai.ChatCompletion.create(
					model="gpt-3.5-turbo",
					messages=messages,
					temperature=0.5,
					max_tokens=3000,
					frequency_penalty=0.0,
					presence_penalty=0.0,
					n=1
				)
				break
			except openai.error.RateLimitError as e:
				if hasattr(e, "response") and "Retry-After" in e.response.headers:
					wait_time = int(e.response.headers["Retry-After"])
				else:
					wait_time = wait_time * 2
				time.sleep(wait_time)
				retry_count += 1
		else:
			print("Max retry attempts reached.")

		return response['choices'][0].message.content
