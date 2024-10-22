
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from task_automation.task_executor import TaskExecutor, CustomLLM
from intent_recognition.intent_classifier import IntentClassifier
from intent_recognition.entity_extractor import EntityExtractor
from dialog_management.dialog_manager import DialogManager
from dialog_management.response_generator import ResponseGenerator
from task_automation.workflow_integrator import WorkflowIntegrator
from utils.helpers import StreamlitResponse


def main():
    st.set_page_config(layout='wide')
    st.title("Smart Assistant: Prompt-Based Data Analysis and Visualization")
    st.markdown('---')

    # Initialize components
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    dialog_manager = DialogManager()
    response_generator = ResponseGenerator()
    task_executor = TaskExecutor()
    workflow_integrator = WorkflowIntegrator()

    # File uploader
    upload_csv_file = st.file_uploader("Upload Your CSV file for data analysis and visualization", type=["csv"])

    if upload_csv_file is not None:
        data = pd.read_csv(upload_csv_file)
        data.columns = data.columns.str.upper()
        st.table(data.head(5))
        st.write('Data Uploaded Successfully!')

        st.markdown('---')
        st.write('### Enter Your Analysis or Visualization Request')
        query = st.text_area("Enter your prompt")

        if st.button("Submit"):
            if query:
                with st.spinner("Processing..."):
                    # Classify intent and extract entities
                    intent = intent_classifier.predict(query)
                    entities = entity_extractor.extract_entities(query)

                    # Generate dialog response
                    dialog_response = dialog_manager.handle_message(query)
                    response = response_generator.generate_response(intent)

                    st.write('### Assistant Response:')
                    st.write(response)

                    # Execute task based on intent
                    if intent == "data_analysis":
                        st.write('### Data Analysis Result:')
                        result = task_executor.execute_task(query, data)
                        st.write(result)
                    elif intent == "run_workflow":
                        st.write('### Workflow:')
                        workflow_integrator.run_workflow("data_analysis_workflow")
            else:
                st.warning("Please enter a prompt")

if __name__ == "__main__":
    main()
