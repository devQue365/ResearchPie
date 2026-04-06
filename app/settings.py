'''
Configuration settings for linkedin-scraper
'''
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# Load environment variables from .env file
load_dotenv()


# Define the configuration settings
class Settings(BaseSettings):
    '''
    Application settings
    '''
    # Database Settings (defaults)
    database_host: str = "localhost" # By default set to localhost
    database_port: int = 5432 # Default Postgres port
    database_name: str
    database_user: str = "postgres" # Default user config
    database_password: str

    # Scraping settings (defaults)
    max_results_per_query: int = 100 
    max_pages_per_query: int = 10
    concurrent_requests: int = 3
    scrapeops_api_key: str # For fake user agents

    # Logging settings (defaults)
    log_level: str = "INFO"
    log_file: str = r"program-logs.log"

    # Environment settings (defaults)
    debug: bool = True
    environment: str = "development"


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f'''
        SETTINGS :-
        DATABASE SETTINGS :-
        host = {settings.database_host}
        port = {settings.database_port}
        user = {settings.database_user}
        password = {settings.database_password}
       \n\n
        SCRAPING SETTINGS :-
        max_concurrent_requests = {settings.concurrent_requests}
        max_results_per_query = {settings.max_results_per_query}
        max_pages_per_query = {settings.max_pages_per_query}
        
        \n\n
        ENVIRONMENT SETTINGS :-
        debug = {settings.debug}
        environment = {settings.environment}
        \n\n
        ADDITIONAL INFO :-
        root_directory = {settings.root_dir}
        log_directory = {settings.log_dir}
        data directory = {settings.data_dir}
    '''
    @property
    def calculate_database_url(self) -> str:
        '''
        Dynamically calculate database url
        '''
        db_url = f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}"
        return db_url
    
    # Pydantic class config
    class config:
        env_file = ".env"
        case_sensetive = False

# __main__ segment
settings = Settings()