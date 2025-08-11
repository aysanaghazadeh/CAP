import json
import os.path
from LLMs.LLM import LLM
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from util.data.mapping import SENTIMENT_MAP, TOPIC_MAP
from collections import Counter


class PromptGenerator:
    def __init__(self, args):
        self.LLM_model = None
        self.descriptions = None
        self.sentiments = None
        self.topics = None
        self.audiences = None
        self.physical_sensation = None
        self.objects = None
        self.set_LLM(args)
        self.set_descriptions(args)
        self.set_sentiments(args)
        self.set_topics(args)
        self.set_audience(args)
        self.set_physical_sensation(args)
        self.set_objects(args)
    def set_LLM(self, args):
        if args.text_input_type == 'LLM':
            self.LLM_model = LLM(args)

    def set_descriptions(self, args):
        if args.text_input_type not in ['LLM', 'AR']:
            self.descriptions = self.get_all_descriptions(args)

    def set_sentiments(self, args):
        if args.with_sentiment:
            self.sentiments = self.get_all_sentiments(args)

    def set_topics(self, args):
        if args.with_sentiment:
            self.topics = self.get_all_topics(args)

    def set_audience(self, args):
        if args.with_audience:
            self.audiences = self.get_all_audience(args)
    
    def set_physical_sensation(self, args):
        if args.with_physical_sensation:
            self.physical_sensation = self.get_all_physical_sensation(args)

    def set_objects(self, args):
        if args.with_objects:
            self.objects = self.get_all_objects(args)
    
    @staticmethod
    def get_all_sentiments(args):
        if not args.with_sentiment:
            return None
        sentiment_file = os.path.join(args.data_path, 'train/Sentiments_train.json')
        sentiments = json.load(open(sentiment_file))
        return sentiments

    @staticmethod
    def get_all_topics(args):
        if not args.with_topics:
            return None
        topics_file = os.path.join(args.data_path, 'train/Topics_train.json')
        topics = json.load(open(topics_file))
        return topics

    @staticmethod
    def get_all_audience(args):
        if not args.with_audience:
            return None
        audience_file = os.path.join(args.data_path, 'train/get_audience.csv')
        audiences = pd.read_csv(audience_file)
        audiences = audiences.set_index('ID')['description'].to_dict()
        return audiences
    
    @staticmethod
    def get_all_physical_sensation(args):
        if not args.with_physical_sensation:
            return None
        physical_sensation_file = os.path.join(args.data_path, f'train/physical_sensation_prediction_with_definition_{args.LLM}_FTFalse_{args.AD_type}.csv')
        physical_sensations = pd.read_csv(physical_sensation_file)
        physical_sensations = physical_sensations.set_index('ID')['description'].to_dict()
        return physical_sensations
    
    @staticmethod
    def get_all_objects(args):
        if not args.with_objects:
            return None
        objects_file = os.path.join(args.data_path, f'train/sensation_object_retrieval_{args.LLM}_FTFalse_{args.AD_type}.csv')
        objects = pd.read_csv(objects_file)
        objects = objects.set_index('ID')['description'].to_dict()
        return objects

    @staticmethod
    def get_all_descriptions(args):
        if args.text_input_type in ['AR', 'LLM']:
            return None
        descriptions = pd.read_csv(args.description_file)
        return descriptions

    @staticmethod
    def get_description(image_filename, descriptions):
        return descriptions.loc[descriptions['ID'] == image_filename]['description'].values[0]

    @staticmethod
    def get_LLM_input_prompt(args, action_reason, sentiment=None, topic=None, audience=None, physical_sensation=None, objects=None):
        data = {'action_reason': action_reason, 'sentiment': sentiment, 'topic': topic, 'audience': audience, 'physical_sensation': physical_sensation, 'objects': objects}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.llm_prompt)
        output = template.render(**data)
        return output

    @staticmethod
    def get_most_frequent(values):
        tuple_list = [tuple(inner_list) for inner_list in values]
        # Create a Counter object from the tuple list
        counter = Counter(tuple_list)
        # Get the most common tuple
        most_freq_tuple, _ = counter.most_common(1)[0]
        # Convert tuple back to list if necessary
        return list(most_freq_tuple)[0]

    def get_original_description_prompt(self, args, image_filename):
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        topic = ''
        if args.with_topics:
            if image_filename in self.topics:
                topic_ids = self.topics[image_filename]
                topic_id = self.get_most_frequent([topic_ids])
                if topic_id in TOPIC_MAP:
                    topic = TOPIC_MAP[topic_id]
            else:
                print(f'there is no topic for image: {image_filename}')
        audience = ''
        if args.with_audience:
            if image_filename in self.audiences:
                audience = self.audiences[image_filename]
                if len(audience.split(':')) > 1:
                    audience = audience.split(':')[-1].split('-')[-1]
                else:
                    audience = 'everyone'
            else:
                print(f'there is no audience for image: {image_filename}')
        physical_sensation = 'no sensation'
        if args.with_physical_sensation:
            if image_filename in self.physical_sensation:
                physical_sensation = self.physical_sensation[image_filename]
                if len(physical_sensation.split(':')) > 1:
                    physical_sensation = physical_sensation.split(':')[-1].split('-')[-1]
                else:
                    physical_sensation = 'no sensation'
            else:
                print(f'there is no sensation for image: {image_filename}')
        objects = ''
        if args.with_objects:
            if image_filename in self.objects:
                objects = self.objects[image_filename].split(':\n')[-1]
            else:
                print(f'there is no object for image: {image_filename}')
        data = {'action_reason': action_reason,
                'description': self.get_description(image_filename, self.descriptions).split('Description of the image:')[-1],
                'sentiment': sentiment,
                'topic': topic,
                'audience': audience,
                'physical_sensation': physical_sensation,
                'objects': objects}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        return output

    def get_LLM_generated_prompt(self, args, image_filename):
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        topic = ''
        if args.with_topics:
            if image_filename in self.topics:
                topic_ids = self.topics[image_filename]
                topic_id = self.get_most_frequent([topic_ids])
                if topic_id in TOPIC_MAP:
                    topic = TOPIC_MAP[topic_id]
            else:
                print(f'there is no topic for image: {image_filename}')
        audience = ''
        if args.with_audience:
            if image_filename in self.audiences:
                audience = self.audiences[image_filename]
                if len(audience.split(':')) > 1:
                    audience = audience.split(':')[-1].split('-')[-1]
                else:
                    audience = 'everyone'
            else:
                print(f'there is no audience for image: {image_filename}')
        physical_sensation = 'no sensation'
        if args.with_physical_sensation:
            if image_filename in self.physical_sensation:
                physical_sensation = self.physical_sensation[image_filename]
                if len(physical_sensation.split(':')) > 1:
                    physical_sensation = physical_sensation.split(':')[-1].split('-')[-1]
                else:
                    physical_sensation = 'no sensation'
            else:
                print(f'there is no sensation for image: {image_filename}')
        objects = ''
        if args.with_objects:
            if image_filename in self.objects:
                objects = self.objects[image_filename].split(':\n')[-1]
            else:
                print(f'there is no object for image: {image_filename}')
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        # if image_filename not in QA:
        #     return ""
        # action_reason = []
        # for AR in QA[image_filename][1]:
        #     if AR not in QA[image_filename][0]:
        #         action_reason.append(AR)
        #         break
        LLM_input_prompt = self.get_LLM_input_prompt(args, action_reason, sentiment, topic, audience, physical_sensation, objects)
        description = self.LLM_model(LLM_input_prompt)
        if 'Adjective:' in description:
            adjective = description.split('Adjective:')[1]
            description = description.split('Adjective:')[0]
        else:
            adjective = None
        objects  = ''
        if args.with_objects:
            if image_filename in self.objects:
                objects = self.objects[image_filename].split(':\n')[-1]
            else:
                print(f'there is no object for image: {image_filename}')
        data = {'description': description,
                'action_reason': action_reason,
                'adjective': adjective,
                'sentiment': sentiment,
                'topic': topic,
                'audience': audience,
                'physical_sensation': physical_sensation,
                'objects': objects}

        # print('data:', data)
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        print('LLM generated prompt:', output)
        return output

    def get_AR_prompt(self, args, image_filename):
        sentiment = ''
        if args.with_sentiment:
            if image_filename in self.sentiments:
                sentiment_ids = self.sentiments[image_filename]
                sentiment_id = self.get_most_frequent(sentiment_ids)
                if sentiment_id in SENTIMENT_MAP:
                    sentiment = SENTIMENT_MAP[sentiment_id]
            else:
                print(f'there is no sentiment for image: {image_filename}')
        topic = ''
        if args.with_topics:
            if image_filename in self.topics:
                topic_ids = self.topics[image_filename]
                topic_id = self.get_most_frequent([topic_ids])
                if topic_id in TOPIC_MAP:
                    topic = TOPIC_MAP[topic_id]
            else:
                print(f'there is no topic for image: {image_filename}')
        audience = ''
        if args.with_audience:
            if image_filename in self.audiences:
                audience = self.audiences[image_filename]
                if len(audience.split(':')) > 1:
                    audience = audience.split(':')[-1].split('-')[-1]
                else:
                    audience = 'everyone'
            else:
                print(f'there is no audience for image: {image_filename}')
        QA_path = args.test_set_QA if not args.train else args.train_set_QA
        QA_path = os.path.join(args.data_path, QA_path)
        QA = json.load(open(QA_path))
        action_reason = QA[image_filename][0]
        # action_reason = []
        # for AR in QA[image_filename][1]:
        #     if AR not in QA[image_filename][0]:
        #         action_reason.append(AR)
        #         break
        data = {'action_reason': action_reason, 'sentiment': sentiment, 'topic': topic, 'audience': audience}
        env = Environment(loader=FileSystemLoader(args.prompt_path))
        template = env.get_template(args.T2I_prompt)
        output = template.render(**data)
        print('AR prompt:', output)
        return output

    def generate_prompt(self, args, image_filename):
        prompt_generator_name = f'get_{args.text_input_type}_prompt'
        print('method: ', prompt_generator_name)
        if prompt_generator_name == 'get_LLM_prompt':
            prompt_generator_name = 'get_LLM_generated_prompt'
        prompt_generation_method = getattr(self, prompt_generator_name)
        prompt = prompt_generation_method(args, image_filename)
        return prompt
