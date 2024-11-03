# BASIC SET UP OVERVIEW
**1. Environment Setup**

* **Install LangChain Google Vertex AI:**  Ensure you have the necessary packages installed:

   ```bash
   pip install langchain langchain-google-vertexai google-cloud-aiplatform
   ```

* **Authentication:** LangChain leverages the Google Cloud authentication libraries. You have a few options:

   * **Application Default Credentials (ADC):** The most convenient way if you're working on a Google Cloud project.
      * Run `gcloud auth application-default login` to set this up.
        * NEED TO INSTALL gcloud CLI: https://cloud.google.com/sdk/docs/install#deb
      * Make sure you have the necessary permissions on your Google Cloud project to use Vertex AI.

   * **Service Account Key:** If you need more fine-grained control or are working outside a Google Cloud environment.
      * Create a service account key JSON file.
      * Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of this file.

**2. LangChain Code**

Here's how you create a LangChain `ChatVertexAI` instance and use it:

```python
from langchain_google.vertex_ai import ChatVertexAI
from langchain.schema import HumanMessage

chat = ChatVertexAI(
    project_id="your-project-id",
    location="your-location",  # e.g., 'us-central1'
    model_name="gemini-1.5-pro-preview-0409",
    max_output_tokens=1024,
    temperature=0.7,
)

response = chat([HumanMessage(content="What's the weather in Blacksburg, Virginia?")])
print(response.content)
```
- LOCATION: https://cloud.google.com/vertex-ai/docs/general/locations
- - Columbus, Ohio (us-east5)
- - Dallas, Texas (us-south1)
- - Iowa (us-central1)
- - Las Vegas, Nevada (us-west4)
- - Los Angeles, California (us-west2)
- - Moncks Corner, South Carolina (us-east1)
- - Northern Virginia (us-east4)
- - Oregon (us-west1)
- - Salt Lake City, Utah (us-west3)

**Troubleshooting Tips**

* **Credentials Error:** The most common error is related to authentication. Double-check:
    * You have set up ADC correctly or have the `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to the right file.
    * The service account has the required IAM roles for Vertex AI (`Vertex AI User` or similar).
    * You're using the correct project ID and region.

* **API Not Enabled:** Verify that the Vertex AI API is enabled in your Google Cloud project.
* **Model Name:**  Make sure the `model_name` matches the exact name of the Gemini Pro model available in your region.

**Example with API Calls (Optional):**

If you need more flexibility, you can use the Vertex AI Python SDK directly within LangChain:

```python
from langchain.llms import VertexAI
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="your-location")

llm = VertexAI(model_name="gemini-1.5-pro-preview-0409")
response = llm("What's the weather in Blacksburg, Virginia?")
print(response)
```

# Authentication process for accessing Gemini 1.5 Pro on Vertex AI using an Ubuntu machine.

**Overview**

Google Cloud Platform (GCP) uses service accounts to manage authentication and authorization for programmatic access to its services, including Vertex AI. There are two primary ways to set this up on your Ubuntu machine:

1. **Application Default Credentials (ADC):** Recommended for simplicity and ease of use.
2. **Service Account Key File:** Provides more fine-grained control but requires careful management of the key file.

**Method 1: Application Default Credentials (ADC)**

**Set up gcloud cli:**
* SOURCE: https://cloud.google.com/sdk/docs/install#deb

**1. Add the Google Cloud Repository:**

```bash
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
```

This will add Google Cloud's package repository to your system's sources list.

**2. Update Package Lists and Install:**

```bash
sudo apt-get update && sudo apt-get install google-cloud-sdk
```

This will update your package lists and install the Google Cloud SDK.

**3. Verify Installation:**

```bash
gcloud --version
```

You should see the installed version of the Google Cloud SDK.

**Troubleshooting:**

* **Outdated Keys:** If you receive an error about expired keys, you can update them with:

  ```bash
  sudo apt-get install google-cloud-sdk-gpg-keys
  ```

* **Proxy Issues:** If you are behind a proxy, you might need to configure your proxy settings for the `apt` package manager and the `gcloud` CLI. You can find instructions on how to do this in the Google Cloud SDK documentation.

**Continuing with the Authentication Process:**

Now that you have the Google Cloud SDK installed, you can proceed with the authentication steps I mentioned earlier:

1. **Initialize gcloud:**
   ```bash
   gcloud init
   ```

2. **Authenticate:**
   ```bash
   gcloud auth application-default login
   ```

   This will open a browser window for you to authenticate. Once completed, your credentials will be stored in the Application Default Credentials location (`~/.config/gcloud/application_default_credentials.json`).

**How it works:**

* When you use LangChain's `ChatVertexAI`, it automatically looks for credentials in the ADC location.
* If found, it uses these credentials to authenticate your requests to Vertex AI.
* The beauty of ADC is that you don't need to explicitly pass credentials in your code; it's all handled behind the scenes.

3. **gcloud config**
   - You will have to select a config (this lets you manage properties gcloud cli uses)
   - more info: https://cloud.google.com/sdk/gcloud/reference/config
   - You can also use `gcloud config set` to set a specific property
**Method 2: Service Account Key File**

**Steps:**

1. **Create a Service Account:** In your Google Cloud Console, navigate to "IAM & Admin" -> "Service Accounts" and create a new service account.
2. **Grant Permissions:** Assign the "Vertex AI User" role (or a more specific role like "Vertex AI Model User") to the service account.
3. **Create a Key:**  Generate a JSON key file for the service account and download it securely to your Ubuntu machine. 
4. **Set the Environment Variable:** Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the downloaded JSON key file:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
   ```

**How it works:**

* LangChain will look for the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.
* If set, it uses the credentials in the specified JSON file to authenticate with Vertex AI.

**Important Security Considerations**

* **Never share your service account key file publicly!**
* **Store your key file securely:** Consider using a secrets manager or a tool like `keyring` to manage the key file securely.
* **Least Privilege:** Only grant the minimum permissions necessary for the service account.

# view your existing Google Cloud projects and get their IDs using the `gcloud` CLI:

Using Application Default Credentials (ADC) is generally the most convenient way to authenticate. 

**1. List All Projects:**

```bash
gcloud projects list
```

This command will display a table with the following columns:

* `PROJECT_ID`: The unique identifier for the project.
* `NAME`: The display name of the project.
* `PROJECT_NUMBER`: A unique numerical identifier.

**2. Get Current Project's ID:**

If you've already set a default project for your gcloud CLI, you can get its ID with:

```bash
gcloud config get-value project
```

**3. Set a Project as Default:**

To switch between projects or set a default project for your future `gcloud` commands, use:

```bash
gcloud config set project your-project-id
```

Replace `your-project-id` with the actual ID of the project you want to use.

**4. Get Project ID from a Specific Project Name:**

If you know the name of the project, but not its ID, you can filter the output of `gcloud projects list` and extract the ID:

```bash
gcloud projects list --filter="name:your-project-name" --format="value(projectId)"
```

Replace `your-project-name` with the name of the project you're looking for.

**Important Notes:**

* **Project Selection:** Make sure you select the correct project ID that corresponds to the project where you have enabled Vertex AI and where you want to use Gemini 1.5 Pro.
* **Credentials:** After setting the default project, the `gcloud auth application-default login` command should automatically use the credentials associated with that project. If you encounter any authentication errors, double-check that you've logged in with the correct Google account.

**Mark a project for deletion**
1. In the Google Cloud console, go to the IAM & Admin Settings page.
* Go to IAM & Admin Settings

2. On the IAM & Admin Settings page, select the project you want to delete, and then click Shut down.
* Shut down only appears for users who have the resourcemanager.projects.delete permission in the current project.

3. Click Shut down.
4. Enter the project ID, and then click Shut down anyway.
* This method immediately marks a project to be deleted. A notification email is sent to the user who initiated the delete operation and the Technical category contacts that are listed in Essential Contacts on a best effort basis; if the notification fails to send, the project is still marked to be deleted. If there's no contact in the Technical category, the fallback contact isn't notified.

**Delete configuration in cli**
- https://cloud.google.com/sdk/gcloud/reference/config/configurations/delete
- 'you cannot delete an active configuration'