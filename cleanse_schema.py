from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType
from pyspark.sql.functions import col


def replace_spaces_with_underscores(df: DataFrame) -> DataFrame:
    """
    Recursively replace spaces with underscores in field names of a PySpark DataFrame,
    including nested structs, arrays, and maps.

    Args:
        df: Input PySpark DataFrame
    Returns:
        DataFrame with spaces replaced by underscores in field names
    """

    def transform_schema(schema, prefix=""):
        new_fields = []
        for field in schema.fields:
            # Clean field name
            new_name = field.name.replace(" ", "_")

            # Handle different data types
            if isinstance(field.dataType, StructType):
                # Recursively process nested struct
                nested_schema = transform_schema(field.dataType, f"{prefix}{new_name}.")
                new_field = StructField(new_name, nested_schema, field.nullable)
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                # Handle array of structs
                nested_schema = transform_schema(field.dataType.elementType, f"{prefix}{new_name}.")
                new_field = StructField(new_name, ArrayType(nested_schema), field.nullable)
            elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
                # Handle map with struct values
                nested_schema = transform_schema(field.dataType.valueType, f"{prefix}{new_name}.")
                new_field = StructField(new_name, MapType(field.dataType.keyType, nested_schema), field.nullable)
            else:
                # Simple field, just update name
                new_field = StructField(new_name, field.dataType, field.nullable)

            new_fields.append(new_field)

        return StructType(new_fields)

    # Create new schema
    new_schema = transform_schema(df.schema)

    # Create new DataFrame with updated schema
    # We need to select columns with new names
    def build_select_expr(schema, prefix=""):
        exprs = []
        for field in schema.fields:
            old_name = field.name.replace("_", " ")
            new_name = field.name
            full_old_name = f"{prefix}{old_name}"
            full_new_name = f"{prefix}{new_name}"

            if isinstance(field.dataType, StructType):
                # Handle nested struct
                nested_exprs = build_select_expr(field.dataType, f"{full_old_name}.")
                exprs.append(struct(nested_exprs).alias(full_new_name))
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                # Handle array of structs
                nested_exprs = build_select_expr(field.dataType.elementType, "")
                exprs.append(transform(col(full_old_name), lambda x: struct(nested_exprs)).alias(full_new_name))
            else:
                # Simple field
                exprs.append(col(f"`{full_old_name}`").alias(full_new_name))

        return exprs

    from pyspark.sql.functions import struct, transform
    select_expr = build_select_expr(new_schema)

    return df.select(select_expr)


def combine_case_insensitive_duplicates(df: DataFrame) -> DataFrame:
    """
    Combine case-insensitive duplicate fields at the same level in a PySpark DataFrame.
    Keeps the first occurrence of the field (case-insensitive) and coalesces values.

    Args:
        df: Input PySpark DataFrame
    Returns:
        DataFrame with case-insensitive duplicates combined
    """

    def process_schema(schema, prefix=""):
        # Track seen field names (lowercase) and their original names
        seen_fields = {}
        select_exprs = []

        for field in schema.fields:
            field_lower = field.name.lower()
            full_name = f"{prefix}{field.name}"

            if field_lower not in seen_fields:
                # New field, add to tracking
                seen_fields[field_lower] = {
                    'original_name': field.name,
                    'full_name': full_name,
                    'data_type': field.dataType
                }

                if isinstance(field.dataType, StructType):
                    # Recursively process nested struct
                    nested_exprs = process_schema(field.dataType, f"{full_name}.")
                    select_exprs.append(struct(nested_exprs).alias(field.name))
                elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                    # Handle array of structs
                    nested_exprs = process_schema(field.dataType.elementType, "")
                    select_exprs.append(transform(col(full_name), lambda x: struct(nested_exprs)).alias(field.name))
                else:
                    # Simple field
                    select_exprs.append(col(f"`{full_name}`").alias(field.name))
            else:
                # Duplicate field, coalesce with existing
                original_info = seen_fields[field_lower]
                original_name = original_info['original_name']

                if isinstance(field.dataType, StructType):
                    # For structs, we need to merge fields recursively
                    nested_exprs = process_schema(field.dataType, f"{full_name}.")
                    select_exprs = [expr for expr in select_exprs if expr._jc.getAlias() != original_name]
                    select_exprs.append(struct(nested_exprs).alias(original_name))
                else:
                    # For simple fields, use coalesce
                    for i, expr in enumerate(select_exprs):
                        if expr._jc.getAlias() == original_name:
                            select_exprs[i] = coalesce(col(f"`{original_info['full_name']}`"),
                                                       col(f"`{full_name}`")).alias(original_name)
                            break

        return select_exprs

    from pyspark.sql.functions import coalesce, struct, transform
    select_expr = process_schema(df.schema)

    return df.select(select_expr)


# Example usage:
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

spark = SparkSession.builder.appName("Test").getOrCreate()

# Sample schema with spaces and case-insensitive duplicates
schema = StructType([
    StructField("First Name", StringType(), True),
    StructField("first name", StringType(), True),
    StructField("User Info", StructType([
        StructField("Age Group", IntegerType(), True),
        StructField("age group", IntegerType(), True)
    ]), True),
    StructField("Contact Details", ArrayType(
        StructType([
            StructField("Phone Number", StringType(), True)
        ])
    ), True)
])

# Sample data
data = [
    ("John", "Johnny", {"Age Group": 30, "age group": 31}, [{"Phone Number": "123"}]),
    ("Jane", "Janey", {"Age Group": 25, "age group": 26}, [{"Phone Number": "456"}])
]

df = spark.createDataFrame(data, schema)

# Apply functions
df_cleaned = replace_spaces_with_underscores(df)
df_final = combine_case_insensitive_duplicates(df_cleaned)

df_final.printSchema()
"""
