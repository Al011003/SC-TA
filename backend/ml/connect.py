from sqlalchemy import create_engine
import pandas as pd
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    user = "postgres"  # default user di Supabase
    password = os.environ.get("SUPABASE_DB_PASSWORD")  
    host = os.environ.get("SUPABASE_DB_HOST")          
    db = "postgres"                                    

    # encode password supaya aman (jika ada karakter khusus)
    password = quote_plus(password)

    return create_engine(f"postgresql://{user}:{password}@{host}:5432/{db}")

def load_financial_data():
    engine = get_engine()

    query = """
        SELECT 
            tahun,
            kode_perusahaan,
            kuartal,
            NPM,
            revneg,
            netprofneg,
            ihsg,
            lq45,
            netprofit,
            revenue
        FROM NPM;
    """

    df = pd.read_sql(query, engine)
    return df
