
import logfire
logfire.configure()
logfire.instrument_pydantic_ai()

import nest_asyncio

nest_asyncio.apply()
logfire.instrument_httpx(capture_all=True)

import os

from pydantic_ai.agent import Agent

# setup Groq API Key
os.environ["GEMINI_API_KEY"] = "Your_Gemini_API_KEY"




from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
import json

from dataclasses import dataclass

import httpx

from pydantic_ai import Agent


@dataclass
class MyDeps:
    ticket_id : str
    customer_tier : str
    subject: str
    message : str
    previous_tickets: int
    monthly_revenue : float
    account_age_days: int



from pydantic import BaseModel
class Analysis(BaseModel):
    Urgency: str
    Priority: str
    Team:str
    Reason: str

class Safeguard(BaseModel):
    result: str

class Updated(BaseModel):
    updated: str

combined_agent = Agent(
    'google-gla:gemini-2.0-flash',
    system_prompt=(
        f''' You are a Support ticket analyzer and router,
        Based on the information related to the support ticket ,'''
        'Decide the urgency of the issue,Minimum and Maximum time to be taken to resolve the ticket'
        'And decide whether the ticket needs to be routed as high priority, low priority or medium priority.'
        'Also decide whether the ticket needs to be routed to technical team or Developers or Devops team or finance team or business stakeholders or can be solved by chatbot which clears the customer queries and explains the customer about the app'
        'check whether the analysis you did is accepted by the rules by retrieving the rules using rules_retriever'
        'You must output - Urgency: str,Priority: str,Team: str,Reason: str, chatbot_resolvable: str'



    ),output_type=Analysis
)

SafeGaurd_agent = Agent(
    'google-gla:gemini-2.0-flash',
    system_prompt=(
        f''' You are a safeguard, who checks the output given by ticket analyzer which includes the output and reasoning behind it,
         You need to check whether the analyzer was confident about the output or ambiguos,
        if the output is unclear or the reasoning couldn't explain the output properly you would send a message `AMBIGUOS` else `CLEAR` '''),
    output_type=str
)


Learning_and_helping_agent = Agent('google-gla:gemini-2.0-flash',
    system_prompt=(
        f''' You are a learner, Based on the output given by ticket analyser and its corresponding Human feedback,you do the following things:
        Take the existing rules using rules_retriever
        1. Create a set of rules the ticket analyser should follow if there aren't any
        2. add a set of rules for existing rules if the human feedback contains new input
        3. Tweak the existing rules based on human feedback
        4. Do not add duplicate rules, if a rule is already present in the existing rules then do not add it.

        and save it using rules_saver into a file
        If Human is not provided just look at the retrieved riles and check if you can add some thing based on analysers output
        Analyzer does these things:
        takes data :
            ticket_id
            customer_tier
            subject
            message
            previous_tickets
            monthly_revenue
            account_age_days'''
        'Based on the information related to the support ticket calculates importance and analyses text'
        'Decides the urgency of the issue,Minimum and Maximum time to be taken to resolve the ticket'
        'And decides whether the ticket needs to be routed as high priority, low priority or medium priority.'
        '''Also decides whether the ticket needs to be routed to technical team or Developers or Devops team or finance team or business stakeholders or
         can be solved by chatbot which clears the customer queries and explains the customer about the app

         Output : if saved new rules, Yes else no'''
        ),
    output_type=Updated)

@Learning_and_helping_agent.tool
@combined_agent.tool
async def rules_retriever(ctx: RunContext[None], _: str):
  filepath = "rules.txt"
  r = "r"
  with open(filepath, r) as f:
      data = f.read()
  return data

@Learning_and_helping_agent.tool_plain
async def rules_saver(rules: str):
  filepath = "rules.txt"
  a = "w"
  with open(filepath, a) as f:
      f.write(rules)



Human_feedback_agent =  Agent('google-gla:gemini-2.0-flash',
    system_prompt=( f''' You are a human and the safegaurd routes tickets to you when ticket analyzer gives ambiguos results, you need to analyze the output provided by the agent Ticket Analyzer
    and specify the reason for the ambiguity and also specify should have any improvements in the output and give the feedback based on that''')
    ,)


def run_agent(input):

  result = combined_agent.run_sync(input,model_settings={'temperature': 0.0}

  )

  safegaurd_result = SafeGaurd_agent.run_sync(f'This is the result of ticket analyzer {result.output}',model_settings={'temperature': 0.0}

  )
  if safegaurd_result.output == 'CLEAR':
    Learning_and_helping_result = Learning_and_helping_agent.run_sync(f'This is the result of ticket analyzer {result.output}',model_settings={'temperature': 0.0}

    )
    Human_feedback_result = 'Answer is clear, no need for human feedback'
  else:
    print('Human intervention needed')
    Human_feedback_result = Human_feedback_agent.run_sync(f'This is the result of ticket analyzer {result.output}, this was flagged as ambiguous output',model_settings={'temperature': 0.0})
    Learning_and_helping_result = Learning_and_helping_agent.run_sync(f'This is the result of ticket analyzer {result.output} and This is the Humanfeedback: {Human_feedback_result.output}',model_settings={'temperature': 0.0}

    )

  return result,safegaurd_result,Human_feedback_result,Learning_and_helping_result
