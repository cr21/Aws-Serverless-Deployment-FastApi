from typing import Annotated
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fasthtml import FastHTML
from fasthtml.common import (
    Html,
    Script,
    Head,
    Title,
    Body,
    Div,
    Form,
    Input,
    Img,
    P,
    to_xml,
)
from shad4fast import (
    ShadHead,
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    CardFooter,
    Alert,
    AlertTitle,
    AlertDescription,
    Button,
    Badge,
    Separator,
    Lucide,
    Progress,
)
import base64

# Create main FastAPI app
app = FastAPI(
    title="Food Image Classification API",
    description="FastAPI application serving an ONNX model for Food image classification",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
INPUT_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
LABELS = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']

# Load the ONNX model
try:
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("onxx_models/food_101_vit_small/model.onnx")
    ort_session.run(
        ["output"], {"input": np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)}
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: dict  # Change to dict for class probabilities
    success: bool
    message: str

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image for model inference

    Args:
        image (PIL.Image): Input image

    Returns:
        np.ndarray: Preprocessed image array
    """
    # Convert to RGB if not already
    image = image.convert("RGB")

    # Resize
    image = image.resize(INPUT_SIZE)

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply mean and std normalization
    img_array = (img_array - MEAN) / STD

    # Transpose to channel-first format (NCHW)
    img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)

    return img_array


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    content = Html(
        Head(
            Title("Food Image Classifier"),
            ShadHead(tw_cdn=True, theme_handle=True),
            Script(
                src="https://unpkg.com/htmx.org@2.0.3",
                integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                crossorigin="anonymous",
            ),
            Script("""
                htmx.onLoad(function(content) {
                    if (content.id === 'prediction-results') {
                        const predictionRows = content.querySelectorAll('.prediction-row');
                        predictionRows.forEach((row, index) => {
                            const confidence = parseFloat(row.dataset.confidence) * 100;
                            row.querySelector('.confidence-bar').style.width = `${confidence}%`;
                        });
                    }
                });
            """)
        ),
        Body(
            Div(
                Card(
                    CardHeader(
                        Div(
                            CardTitle("Food Image Classifier 🍽️"),
                            Badge("AI Powered", variant="secondary", cls="w-fit"),
                            cls="flex items-center justify-between",
                        ),
                        CardDescription(
                            "Upload a food image to identify its type. Our AI model will analyze it instantly!"
                        ),
                    ),
                    CardContent(
                        Form(
                            Div(
                                Div(
                                    Input(
                                        type="file",
                                        name="file",
                                        accept="image/*",
                                        required=True,
                                        cls="mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:border-primary file:font-semibold file:bg-primary/10 file:text-primary hover:file:bg-primary/20 file:cursor-pointer",
                                    ),
                                    P(
                                        "Drag and drop an image or click to browse",
                                        cls="text-sm text-muted-foreground text-center mt-2",
                                    ),
                                    cls="border-2 border-dashed rounded-lg p-4 hover:border-primary/50 transition-colors",
                                ),
                                Button(
                                    Lucide("sparkles", cls="mr-2 h-4 w-4"),
                                    "Classify Image",
                                    type="submit",
                                    cls="w-full mt-4",
                                ),
                                Div(id="prediction-results", cls="mt-6"),
                                cls="space-y-4",
                            ),
                            enctype="multipart/form-data",
                            hx_post="/classify",
                            hx_target="#prediction-results",
                        ),
                    ),
                    cls="w-full max-w-3xl shadow-lg",
                    standard=True,
                ),
                cls="container flex items-center justify-center min-h-screen p-4 bg-background",
            ),
            cls="bg-background text-foreground",
        ),
    )
    return to_xml(content)

# FastAPI routes
# @app.get("/", response_class=HTMLResponse)
# async def ui_home():
#     content = Html(
#         Head(
#             Title("Cat vs Dog Classifier"),
#             ShadHead(tw_cdn=True, theme_handle=True),
#             Script(
#                 src="<https://unpkg.com/htmx.org@2.0.3>",
#                 integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
#                 crossorigin="anonymous",
#             ),
#         ),
#         Body(
#             Div(
#                 Card(
#                     CardHeader(
#                         Div(
#                             CardTitle("Cat vs Dog Classifier 🐱 🐶"),
#                             Badge("AI Powered", variant="secondary", cls="w-fit"),
#                             cls="flex items-center justify-between",
#                         ),
#                         CardDescription(
#                             "Upload an image to classify whether it's a cat or a dog. Our AI model will analyze it instantly!"
#                         ),
#                     ),
#                     CardContent(
#                         Form(
#                             Div(
#                                 Div(
#                                     Input(
#                                         type="file",
#                                         name="file",
#                                         accept="image/*",
#                                         required=True,
#                                         cls="mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 file:cursor-pointer",
#                                     ),
#                                     P(
#                                         "Drag and drop an image or click to browse",
#                                         cls="text-sm text-muted-foreground text-center mt-2",
#                                     ),
#                                     cls="border-2 border-dashed rounded-lg p-4 hover:border-primary/50 transition-colors",
#                                 ),
#                                 Button(
#                                     Lucide("sparkles", cls="mr-2 h-4 w-4"),
#                                     "Classify Image",
#                                     type="submit",
#                                     cls="w-full",
#                                 ),
#                                 cls="space-y-4",
#                             ),
#                             enctype="multipart/form-data",
#                             hx_post="/classify",
#                             hx_target="#result",
#                         ),
#                         Div(id="result", cls="mt-6"),
#                     ),
#                     cls="w-full max-w-3xl shadow-lg",
#                     standard=True,
#                 ),
#                 cls="container flex items-center justify-center min-h-screen p-4",
#             ),
#             cls="bg-background text-foreground",
#         ),
#     )
#     return to_xml(content)

@app.post("/classify", response_class=HTMLResponse)
async def ui_handle_classify(file: Annotated[bytes, File()]):
    try:
        response = await predict(file)
        image_b64 = base64.b64encode(file).decode("utf-8")

        # Sort predictions by confidence
        sorted_predictions = sorted(response.predictions.items(), key=lambda x: x[1], reverse=True)[:5]

        # Generate HTML for predictions
        prediction_html = Div(
            Div(
                Div(cls="absolute inset-y-0 left-0 bg-primary/20 opacity-50 z-0 confidence-bar"),
                Div(
                    f"{pred[0].replace('_', ' ').title()}",
                    cls="relative z-10 text-sm font-medium"
                ),
                Div(
                    f"{pred[1]*100:.2f}%",
                    cls="text-xs text-muted-foreground"
                ),
                cls=f"prediction-row relative flex justify-between items-center p-2 border-b last:border-b-0 hover:bg-secondary/20 transition-colors",
                data_confidence=str(pred[1])
            ) for pred in sorted_predictions
        )

        # Include the original image
        image_preview = Div(
            Img(
                src=f"data:image/jpeg;base64,{image_b64}",
                cls="max-w-full max-h-64 object-contain rounded-lg mx-auto mb-4"
            ),
            cls="mb-4"
        )

        return to_xml(Div(
            image_preview,
            Div(
                Div("Top 5 Predictions", cls="text-lg font-semibold mb-3"),
                prediction_html,
                cls="bg-background border rounded-lg shadow-sm"
            )
        ))
    except Exception as e:
        return to_xml(Div(
            f"Error processing image: {str(e)}",
            cls="text-red-500 p-4 bg-red-50 rounded-lg"
        ))

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: Annotated[bytes, File(description="Image file to classify")]):
    try:
        image = Image.open(io.BytesIO(file))
        processed_image = preprocess_image(image)

        outputs = ort_session.run(
            ["output"], {"input": processed_image.astype(np.float32)}
        )

        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))

        predictions = {LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}

        return PredictionResponse(
            predictions=predictions, success=True, message="Classification successful"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": True}, status_code=200
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)