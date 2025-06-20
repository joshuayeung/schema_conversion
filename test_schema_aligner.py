from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType
from pyspark.sql.functions import lit, col, struct, array, map_from_arrays
from schema_aligner import align_schema  # Assuming the align_schema function is in schema_aligner.py

# Initialize SparkSession
spark = SparkSession.builder.appName("TestSchemaAligner").getOrCreate()

def test_case_1_simple_fields():
    """Test Case 1: Add simple missing fields"""
    # Source DataFrame schema
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])
    
    # Source DataFrame
    data = [("Alice", 30), ("Bob", 25)]
    df = spark.createDataFrame(data, source_schema)
    
    # Target schema with additional fields
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("city", StringType(), True),
        StructField("country", StringType(), True)
    ])
    
    # Apply schema alignment
    result_df = align_schema(df, target_schema)
    
    # Expected schema
    expected_schema = target_schema
    
    # Verify
    assert result_df.schema == expected_schema, "Schema does not match target schema"
    result_data = result_df.collect()
    expected_data = [
        ("Alice", 30, None, None),
        ("Bob", 25, None, None)
    ]
    assert [row.asDict() for row in result_data] == [
        {"name": name, "age": age, "city": city, "country": country}
        for name, age, city, country in expected_data
    ], "Data does not match expected output"
    print("Test Case 1: Passed")

def test_case_2_nested_struct():
    """Test Case 2: Add fields in nested struct"""
    # Source DataFrame schema
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("info", StructType([
            StructField("age", IntegerType(), True)
        ]), True)
    ])
    
    # Source DataFrame
    data = [("Alice", {"age": 30}), ("Bob", {"age": 25})]
    df = spark.createDataFrame(data, source_schema)
    
    # Target schema with additional nested fields
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("info", StructType([
            StructField("age", IntegerType(), True),
            StructField("city", StringType(), True)
        ]), True),
        StructField("country", StringType(), True)
    ])
    
    # Apply schema alignment
    result_df = align_schema(df, target_schema)
    
    # Expected schema
    expected_schema = target_schema
    
    # Verify
    assert result_df.schema == expected_schema, "Schema does not match target schema"
    result_data = result_df.collect()
    expected_data = [
        ("Alice", {"age": 30, "city": None}, None),
        ("Bob", {"age": 25, "city": None}, None)
    ]
    assert [row.asDict(recursive=True) for row in result_data] == [
        {"name": name, "info": info, "country": country}
        for name, info, country in expected_data
    ], "Data does not match expected output"
    print("Test Case 2: Passed")

def test_case_3_array_of_structs():
    """Test Case 3: Add fields in array of structs"""
    # Source DataFrame schema
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("scores", ArrayType(
            StructType([
                StructField("subject", StringType(), True)
            ])
        ), True)
    ])
    
    # Source DataFrame
    data = [
        ("Alice", [{"subject": "Math"}, {"subject": "Science"}]),
        ("Bob", [{"subject": "History"}])
    ]
    df = spark.createDataFrame(data, source_schema)
    
    # Target schema with additional fields in array struct
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("scores", ArrayType(
            StructType([
                StructField("subject", StringType(), True),
                StructField("grade", IntegerType(), True)
            ])
        ), True)
    ])
    
    # Apply schema alignment
    result_df = align_schema(df, target_schema)
    
    # Expected schema
    expected_schema = target_schema
    
    # Verify
    assert result_df.schema == expected_schema, "Schema does not match target schema"
    result_data = result_df.collect()
    expected_data = [
        ("Alice", [{"subject": "Math", "grade": None}, {"subject": "Science", "grade": None}]),
        ("Bob", [{"subject": "History", "grade": None}])
    ]
    assert [row.asDict(recursive=True) for row in result_data] == [
        {"name": name, "scores": scores}
        for name, scores in expected_data
    ], "Data does not match expected output"
    print("Test Case 3: Passed")

def test_case_4_map_with_structs():
    """Test Case 4: Add fields in map with struct values"""
    # Source DataFrame schema
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("details", MapType(
            StringType(),
            StructType([
                StructField("value", IntegerType(), True)
            ])
        ), True)
    ])
    
    # Source DataFrame
    data = [
        ("Alice", {"k1": {"value": 10}, "k2": {"value": 20}}),
        ("Bob", {"k1": {"value": 15}})
    ]
    df = spark.createDataFrame(data, source_schema)
    
    # Target schema with additional fields in map struct
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("details", MapType(
            StringType(),
            StructType([
                StructField("value", IntegerType(), True),
                StructField("category", StringType(), True)
            ])
        ), True)
    ])
    
    # Apply schema alignment
    result_df = align_schema(df, target_schema)
    
    # Expected schema
    expected_schema = target_schema
    
    # Verify
    assert result_df.schema == expected_schema, "Schema does not match target schema"
    result_data = result_df.collect()
    expected_data = [
        ("Alice", {"k1": {"value": 10, "category": None}, "k2": {"value": 20, "category": None}}),
        ("Bob", {"k1": {"value": 15, "category": None}})
    ]
    assert [row.asDict(recursive=True) for row in result_data] == [
        {"name": name, "details": details}
        for name, details in expected_data
    ], "Data does not match expected output"
    print("Test Case 4: Passed")

def test_case_5_complex_nested():
    """Test Case 5: Complex nested structure with structs, arrays, and maps"""
    # Source DataFrame schema
    source_schema = StructType([
        StructField("name", StringType(), True),
        StructField("profile", StructType([
            StructField("scores", ArrayType(
                StructType([
                    StructField("subject", StringType(), True)
                ])
            ), True),
            StructField("metadata", MapType(
                StringType(),
                StructType([
                    StructField("value", IntegerType(), True)
                ])
            ), True)
        ]), True)
    ])
    
    # Source DataFrame
    data = [
        ("Alice", {
            "scores": [{"subject": "Math"}, {"subject": "Science"}],
            "metadata": {"k1": {"value": 100}}
        })
    ]
    df = spark.createDataFrame(data, source_schema)
    
    # Target schema with additional fields
    target_schema = StructType([
        StructField("name", StringType(), True),
        StructField("profile", StructType([
            StructField("scores", ArrayType(
                StructType([
                    StructField("subject", StringType(), True),
                    StructField("grade", IntegerType(), True)
                ])
            ), True),
            StructField("metadata", MapType(
                StringType(),
                StructType([
                    StructField("value", IntegerType(), True),
                    StructField("category", StringType(), True)
                ])
            ), True),
            StructField("city", StringType(), True)
        ]), True),
        StructField("country", StringType(), True)
    ])
    
    # Apply schema alignment
    result_df = align_schema(df, target_schema)
    
    # Expected schema
    expected_schema = target_schema
    
    # Verify
    assert result_df.schema == expected_schema, "Schema does not match target schema"
    result_data = result_df.collect()
    expected_data = [
        ("Alice", {
            "scores": [{"subject": "Math", "grade": None}, {"subject": "Science", "grade": None}],
            "metadata": {"k1": {"value": 100, "category": None}},
            "city": None
        }, None)
    ]
    assert [row.asDict(recursive=True) for row in result_data] == [
        {"name": name, "profile": profile, "country": country}
        for name, profile, country in expected_data
    ], "Data does not match expected output"
    print("Test Case 5: Passed")

# Run all test cases
try:
    test_case_1_simple_fields()
    test_case_2_nested_struct()
    test_case_3_array_of_structs()
    test_case_4_map_with_structs()
    test_case_5_complex_nested()
    print("All test cases passed successfully!")
except AssertionError as e:
    print(f"Test failed: {str(e)}")
finally:
    spark.stop()
