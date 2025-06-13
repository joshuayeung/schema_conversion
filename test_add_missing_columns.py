import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType, MapType, LongType, DecimalType
)
from pyspark.sql.functions import col
from add_missing_columns import add_missing_columns
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.types import (
    NestedField, IntegerType, StringType, StructType, ListType, MapType, DecimalType
)

@pytest.fixture(scope="module")
def spark():
    """Fixture to create a SparkSession."""
    spark = SparkSession.builder.appName("TestAddMissingColumns").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def simple_df(spark):
    """Fixture for a simple DataFrame with name and age."""
    data = [("Alice", 30), ("Bob", 25)]
    return spark.createDataFrame(data, ["name", "age"])

@pytest.fixture
def nested_df(spark):
    """Fixture for a DataFrame with a nested struct."""
    data = [
        ("Alice", {"street": "123 Main"}),
        ("Bob", {"street": "456 Oak"})
    ]
    return spark.createDataFrame(data, ["name", "address"])

@pytest.fixture
def target_schema():
    """Fixture for a target Iceberg schema with additional fields."""
    return IcebergSchema(
        NestedField(field_id=1, name="id", field_type=IntegerType(), required=True),
        NestedField(field_id=2, name="name", field_type=StringType(), required=False),
        NestedField(field_id=3, name="age", field_type=IntegerType(), required=False),
        NestedField(field_id=4, name="price", field_type=DecimalType(10, 2), required=False),
        NestedField(field_id=5, name="address", field_type=StructType([
            NestedField(field_id=6, name="street", field_type=StringType(), required=False),
            NestedField(field_id=7, name="city", field_type=StringType(), required=False)
        ]), required=False),
        NestedField(field_id=8, name="tags", field_type=ListType(StringType(), element_required=False), required=False),
        NestedField(field_id=9, name="attributes", field_type=MapType(
            StringType(), StructType([
                NestedField(field_id=10, name="value", field_type=IntegerType(), required=False),
                NestedField(field_id=11, name="category", field_type=StringType(), required=False)
            ]), value_required=False), required=False)
    )

def test_add_missing_primitive_columns(spark, simple_df, target_schema):
    """Test adding missing primitive columns (id, price)."""
    from iceberg_to_pyspark_schema import iceberg_to_pyspark_schema
    target_spark_schema = iceberg_to_pyspark_schema(target_schema)
    
    result_df = add_missing_columns(simple_df, target_spark_schema)
    
    # Verify schema
    assert result_df.schema == target_spark_schema
    
    # Verify data
    result_data = result_df.select("name", "age", "id", "price").collect()
    assert len(result_data) == 2
    assert result_data[0]["name"] == "Alice" and result_data[0]["age"] == 30
    assert result_data[0]["id"] is None and result_data[0]["price"] is None
    assert result_data[1]["name"] == "Bob" and result_data[1]["age"] == 25
    assert result_data[1]["id"] is None and result_data[1]["price"] is None

def test_add_missing_struct_column(spark, simple_df, target_schema):
    """Test adding missing struct column (address)."""
    from iceberg_to_pyspark_schema import iceberg_to_pyspark_schema
    target_spark_schema = iceberg_to_pyspark_schema(target_schema)
    
    result_df = add_missing_columns(simple_df, target_spark_schema)
    
    # Verify address struct
    address_field = next(f for f in result_df.schema.fields if f.name == "address")
    assert isinstance(address_field.dataType, StructType)
    assert len(address_field.dataType.fields) == 2
    assert address_field.dataType.fields[0].name == "street" and isinstance(address_field.dataType.fields[0].dataType, StringType)
    assert address_field.dataType.fields[1].name == "city" and isinstance(address_field.dataType.fields[1].dataType, StringType)
    
    # Verify address is null
    result_data = result_df.select("address").collect()
    assert all(row["address"] is None for row in result_data)

def test_add_missing_array_column(spark, simple_df, target_schema):
    """Test adding missing array column (tags)."""
    from iceberg_to_pyspark_schema import iceberg_to_pyspark_schema
    target_spark_schema = iceberg_to_pyspark_schema(target_schema)
    
    result_df = add_missing_columns(simple_df, target_spark_schema)
    
    # Verify tags array
    tags_field = next(f for f in result_df.schema.fields if f.name == "tags")
    assert isinstance(tags_field.dataType, ArrayType)
    assert isinstance(tags_field.dataType.elementType, StringType)
    
    # Verify tags is null
    result_data = result_df.select("tags").collect()
    assert all(row["tags"] is None for row in result_data)

def test_add_missing_map_column(spark, simple_df, target_schema):
    """Test adding missing map column with struct values (attributes)."""
    from iceberg_to_pyspark_schema import iceberg_to_pyspark_schema
    target_spark_schema = iceberg_to_pyspark_schema(target_schema)
    
    result_df = add_missing_columns(simple_df, target_spark_schema)
    
    # Verify attributes map
    attributes_field = next(f for f in result_df.schema.fields if f.name == "attributes")
    assert isinstance(attributes_field.dataType, MapType)
    assert isinstance(attributes_field.dataType.keyType, StringType)
    assert isinstance(attributes_field.dataType.valueType, StructType)
    assert len(attributes_field.dataType.valueType.fields) == 2
    assert attributes_field.dataType.valueType.fields[0].name == "value" and isinstance(attributes_field.dataType.valueType.fields[0].dataType, IntegerType)
    assert attributes_field.dataType.valueType.fields[1].name == "category" and isinstance(attributes_field.dataType.valueType.fields[1].dataType, StringType)
    
    # Verify attributes is null
    result_data = result_df.select("attributes").collect()
    assert all(row["attributes"] is None for row in result_data)

def test_no_missing_columns(spark):
    """Test when DataFrame schema matches target schema."""
    data = [("Alice", 30)]
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])
    df = spark.createDataFrame(data, schema)
    
    result_df = add_missing_columns(df, schema)
    
    # Verify schema and data unchanged
    assert result_df.schema == schema
    result_data = result_df.collect()
    assert len(result_data) == 1
    assert result_data[0]["name"] == "Alice" and result_data[0]["age"] == 30

def test_nested_struct_partial_match(spark, nested_df):
    """Test updating nested struct with missing field (city)."""
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("address", StructType([
            StructField("street", StringType(), True),
            StructField("city", StringType(), True)
        ]), True)
    ])
    
    result_df = add_missing_columns(nested_df, target_schema)
    
    # Verify schema
    assert result_df.schema == target_schema
    
    # Verify data
    result_data = result_df.select("name", "address.street", "address.city").collect()
    assert result_data[0]["name"] == "Alice"
    assert result_data[0]["address.street"] == "123 Main"
    assert result_data[0]["address.city"] is None
    assert result_data[1]["name"] == "Bob"
    assert result_data[1]["address.street"] == "456 Oak"
    assert result_data[1]["address.city"] is None

def test_empty_dataframe(spark, target_schema):
    """Test handling an empty DataFrame."""
    from iceberg_to_pyspark_schema import iceberg_to_pyspark_schema
    target_spark_schema = iceberg_to_pyspark_schema(target_schema)
    
    df = spark.createDataFrame([], StructType([]))
    result_df = add_missing_columns(df, target_spark_schema)
    
    # Verify schema
    assert result_df.schema == target_spark_schema
    
    # Verify no rows
    assert result_df.count() == 0

def test_array_with_nested_struct(spark):
    """Test updating array with nested structs."""
    data = [("Alice", [{"value": 1}])]
    df_schema = StructType([
        StructField("name", StringType(), True),
        StructField("attributes", ArrayType(StructType([
            StructField("value", IntegerType(), True)
        ])), True)
    ])
    df = spark.createDataFrame(data, df_schema)
    
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("attributes", ArrayType(StructType([
            StructField("value", IntegerType(), True),
            StructField("category", StringType(), True)
        ])), True)
    ])
    
    result_df = add_missing_columns(df, target_schema)
    
    # Verify schema
    assert result_df.schema == target_schema
    
    # Verify data
    result_data = result_df.select("name", "attributes").collect()
    assert result_data[0]["name"] == "Alice"
    assert len(result_data[0]["attributes"]) == 1
    assert result_data[0]["attributes"][0]["value"] == 1
    assert result_data[0]["attributes"][0]["category"] is None

def test_map_with_nested_struct(spark):
    """Test updating map with nested structs."""
    data = [("Alice", {"key1": {"value": 1}})]
    df_schema = StructType([
        StructField("name", StringType(), True),
        StructField("attributes", MapType(StringType(), StructType([
            StructField("value", IntegerType(), True)
        ])), True)
    ])
    df = spark.createDataFrame(data, df_schema)
    
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("attributes", MapType(StringType(), StructType([
            StructField("value", IntegerType(), True),
            StructField("category", StringType(), True)
        ])), True)
    ])
    
    result_df = add_missing_columns(df, target_schema)
    
    # Verify schema
    assert result_df.schema == target_schema
    
    # Verify data
    result_data = result_df.select("name", "attributes").collect()
    assert result_data[0]["name"] == "Alice"
    assert result_data[0]["attributes"]["key1"]["value"] == 1
    assert result_data[0]["attributes"]["key1"]["category"] is None
    