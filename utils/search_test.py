import lancedb

# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "../data/lancedb"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("lance_neur_papers_db")


# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------

result = table.search(query="What's docling?", query_type="vector").limit(3)
result.to_pandas()
print(result.to_df())