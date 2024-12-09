import os
from pathlib import Path
from constructs import Construct
from aws_cdk import App, Stack, Environment, Duration, CfnOutput
from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    Architecture,
    FunctionUrlAuthType,
)

# environment setup like what region we should create resouce or stack in
my_environment = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]
)

class FoodImageClassifierFastApiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        # Build Docker image from Dockerfile in the current directory.
        # The Dockerfile should contain necessary dependencies and commands to install gradio and run the application.
        # Note: This assumes you have Docker installed and available in your system.
        # Also, replace 'gradio' with your actual gradio application code.

        # Create stack and deploy Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "FoodImageClassifierFastApi",
            code=DockerImageCode.from_image_asset(str(Path.cwd()), file="Dockerfile"),
            architecture=Architecture.X86_64,
            memory_size=1536,  # 1.5GB memory
            timeout=Duration.minutes(5), # 5 minutes timeout for the Lambda function
        )

        # Add HTTPS URL
        # If we want to expose the Lambda function via HTTPS, we can uncomment the following lines.
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        CfnOutput(self, "functionUrl", value=fn_url.url)

app = App()
gradio_lambda = FoodImageClassifierFastApiStack(app, "FoodImageClassifierFastApiStack", env=my_environment)
app.synth()