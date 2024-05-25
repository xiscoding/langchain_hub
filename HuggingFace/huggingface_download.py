import subprocess

def install_huggingface_cli():
  """
  Installs the Hugging Face CLI if not already installed.
  """
  try:
    subprocess.run(["pip", "install", "huggingface_hub[cli]"])
    print("Hugging Face CLI installed successfully!")
  except subprocess.CalledProcessError:
    print("Error installing Hugging Face CLI. Please install manually using pip install huggingface_hub[cli]")

def download_model(model_id, include_pattern, local_dir):
  """
  Downloads the original weights for a model using the Hugging Face CLI.

  Args:
    model_id: The ID of the model on Hugging Face Hub (e.g., "meta-llama/Meta-Llama-3-8B-Instruct").
    include_pattern: A globbing pattern to specify files to download (e.g., "original/*").
    local_dir: The local directory to save the downloaded files.
  
  ISSUES: symbolics links may become broken paths.
  - use symlinks false
  - Download the actual files from hugging face if needed
  """
  #command = f"huggingface-cli download {model_id} --include {include_pattern} --local-dir {local_dir}"
  command = f"huggingface-cli download {model_id} --local-dir {local_dir} --local-dir-use-symlinks {False}"
  try:
    subprocess.run(command.split(), check=True)
    print(f"Model weights downloaded to {local_dir}")
  except subprocess.CalledProcessError:
    print(f"Error downloading model. Please check the model ID and network connectivity.")

if __name__ == "__main__":
  # Replace with your desired model ID
  #model_id = "LoneStriker/Meta-Llama-3-8B-Instruct-8.0bpw-h8-exl2"

  model_id = "astronomer-io/Llama-3-8B-Instruct-GPTQ-8-Bit"
  include_pattern = "original/*"
  local_dir = "Llama-3-8B-Instruct-GPTQ-8-Bit"
  try:
      subprocess.run(["huggingface-cli", "env"], check=True, capture_output=True, text=True)
  except subprocess.SubprocessError as e:
      print("Hugging Face CLI not found or another error occurred. Trying to install...")
      install_huggingface_cli()

  download_model(model_id, include_pattern, local_dir)
