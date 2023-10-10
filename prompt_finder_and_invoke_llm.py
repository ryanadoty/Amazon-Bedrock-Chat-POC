import os
# from dotenv import load_dotenv
import boto3
import json
import botocore.config
import yaml
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import Bedrock
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma

boto3.setup_default_session(profile_name='bedrock')
config = botocore.config.Config(connect_timeout=120, read_timeout=120)
bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                       config=config)


def load_samples():
    """
    Load the sql examples for few-shot prompting examples
    :return: The sql samples in from the moma_examples.yaml file
    """
    generic_samples = None

    with open("sample_prompts/generic_samples.yaml", "r") as stream:
        generic_samples = yaml.safe_load(stream)
    return generic_samples


def chat_history_loader():
    with open("chat_history.txt", "r") as file:
        chat_history = file.read()
        if chat_history == "":
            return None
        else:
            return chat_history


def prompt_finder(question):
    examples = load_samples()

    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        local_embeddings,
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        k=3
    )

    example_prompt = PromptTemplate(input_variables=["input", "answer"], template="\n\nHuman: {"
                                                                                  "input}\n\nAssistant: {answer}")
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix=f"Chat History: {chat_history_loader()}\n\n" + "Human: {input}\n\nAssistant:",
        input_variables=["input"]
    )

    question_with_prompt = prompt.format(input=question)
    print(question_with_prompt)
    return llm_answer_generator(question_with_prompt)


def llm_answer_generator(question_with_prompt):
    body = json.dumps({"prompt": question_with_prompt,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })
    modelId = 'anthropic.claude-v2'
    # modelId = 'anthropic.claude-instant-v1'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body,
                                    modelId=modelId,
                                    accept=accept,
                                    contentType=contentType)

    response_body = json.loads(response.get('body').read())
    answer = response_body.get('completion')
    return answer
