# Prompt-Based-ERP-Chatbot
This is a Prompt based Data Analysis and Visualization Chatbot for Enterprise Resource Planning. [ERP]

An AI-powered ERP system built with Streamlit for an FMCG company, featuring predictive maintenance, inventory optimization, NLP chatbot, supplier management, role-based access control, data visualization dashboards, and audit logging.

# Features
Predictive Maintenance: Utilizes machine learning models to predict equipment failures, helping to schedule maintenance proactively and reduce downtime.

Inventory Optimization: Implements ARIMA time-series forecasting to predict future inventory needs, optimizing stock levels and minimizing holding costs.

NLP Chatbot: An interactive chatbot powered by spaCy, allowing users to query the ERP system using natural language for tasks like retrieving maintenance logs, summarizing activities, and generating visualizations.

Supplier Management: Enables adding, updating, and deleting supplier information, streamlining procurement processes.

Role-Based Access Control: Ensures secure access to features based on user roles (Admin, Manager, Technician, Employee), maintaining data integrity and confidentiality.

Data Visualization Dashboards: Provides interactive dashboards for maintenance trends, inventory levels, and operational efficiency, with options to download reports as PDFs.

Audit Logging: Tracks and logs user activities within the system, offering an audit trail accessible to Admin users.

# Technologies Used
Streamlit: For building the interactive web application interface.

Pandas: For data manipulation and handling Excel data integration.

NumPy: For numerical computations.

Matplotlib & Plotly: For data visualization and creating interactive graphs.

Scikit-Learn: For implementing machine learning models in predictive maintenance.

Statsmodels: For ARIMA modeling in inventory forecasting.

spaCy: For natural language processing in the chatbot feature.

OpenPyXL: For reading and writing Excel files.

# Getting Started
Prerequisites:

Python 3.7 or higher
Required Python libraries (see requirements.txt)
Installation:

Clone the repository.
Install the required libraries using pip install -r requirements.txt.
Ensure the ERP_project.xlsx file is placed in the project directory.
Running the Application:

Execute streamlit run app.py in the terminal.
Access the application via the provided local URL.
# Usage
Login: Users must log in with their credentials to access the system.
Navigation: Use the sidebar to navigate between different features based on your user role.
Data Input: Authorized users can input maintenance data and manage suppliers directly through the application.
Analytics: Access dashboards and run predictive models to gain insights into operations.
Chatbot: Interact with the NLP chatbot to retrieve information and generate reports using natural language queries.
# Note
This project is intended for educational purposes to demonstrate the integration of AI and machine learning into an ERP system within an FMCG context.
