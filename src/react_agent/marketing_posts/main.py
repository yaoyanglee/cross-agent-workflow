#!/usr/bin/env python
import sys
import json
# from marketing_posts.crew import MarketingPostsCrew
from .crew import MarketingPostsCrew
from dotenv import load_dotenv

load_dotenv()


def run(inputs):
    '''
    Accepts a dictionary as a parameter of the form.

    inputs = {
        'customer_domain': str
        'project_description': str
    }
    '''

    crew_output = MarketingPostsCrew().crew().kickoff(inputs=inputs)
    print("######## CLEAN CREW OUTPUT ###############",
          json.dumps(crew_output.json_dict, indent=2))
    return json.dumps(crew_output.json_dict, indent=2)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'customer_domain': 'crewai.com',
        'project_description': """
CrewAI, a leading provider of multi-agent systems, aims to revolutionize marketing automation for its enterprise clients. This project involves developing an innovative marketing strategy to showcase CrewAI's advanced AI-driven solutions, emphasizing ease of use, scalability, and integration capabilities. The campaign will target tech-savvy decision-makers in medium to large enterprises, highlighting success stories and the transformative potential of CrewAI's platform.

Customer Domain: AI and Automation Solutions
Project Overview: Creating a comprehensive marketing campaign to boost awareness and adoption of CrewAI's services among enterprise clients.
"""
    }
    try:
        MarketingPostsCrew().crew().train(
            n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


if __name__ == "__main__":
    run()
