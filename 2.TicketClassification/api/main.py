# main.py
"""
Entry point to run the Ticket Classification FastAPI server
Professional package-friendly version
"""

import uvicorn
from .app import app  # relative import since api is a package

if __name__ == "__main__":
    # Run the server locally for development
    # Pass the app object directly to uvicorn
    uvicorn.run(
        "api.app:app",              # pass FastAPI instance directly
        host="127.0.0.1", # localhost
        port=8000,        # default FastAPI port
        reload=True       # hot reload on code changes
    )

