from database import engine, Base

def create_database():
    Base.metadata.create_all(engine)
    print("Database successfully created!")

if __name__ == "__main__":
    create_database()
