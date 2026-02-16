# Document Q&A service

## How to Run
1. **Build and run the image**: `docker-compose up --build`  
   *(Note: This will export the ONNX models and run the unit tests: if any fails, the build stops.)*  
2. **To stop the service:** `docker-compose down`  
    *(The uploaded data stays safe in the `./data` folder)*
3. **Documentation**: Visit `http://localhost:8000/docs` for the Swagger UI.

## Endpoints
- `POST /upload`: uploads a .txt or .pdf file to the service.                                
- `POST /ask`: sends a JSON body `{"question": "your question"}` and returns info based on the top relevant stored file.

## Testing
The tests folder contains the unit tests file `test_main.py` which is run at the build stage.
