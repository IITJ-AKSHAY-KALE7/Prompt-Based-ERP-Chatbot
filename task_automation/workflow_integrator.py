import streamlit as st

class WorkflowIntegrator:
    def __init__(self):
        pass

    def create_workflow(self, steps):
        for step in steps:
            if step['type'] == 'text_input':
                st.text_input(step['label'], key=step['key'])
            elif step['type'] == 'button':
                st.button(step['label'], key=step['key'])
            elif step['type'] == 'display':
                st.write(step['content'])

    def run_workflow(self, workflow_name):
        st.title(f"Workflow: {workflow_name}")
        self.create_workflow(self.get_workflow_steps(workflow_name))

    def get_workflow_steps(self, workflow_name):
        return [
            {'type': 'text_input', 'label': 'Enter your name', 'key': 'name'},
            {'type': 'button', 'label': 'Submit', 'key': 'submit'},
            {'type': 'display', 'content': 'Thank you for submitting!'}
        ]