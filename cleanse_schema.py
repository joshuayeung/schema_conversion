from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType
from pyspark.sql.functions import col, struct, transform


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
        name_mapping = {}  # Track original to new names with full paths
        for field in schema.fields:
            # Store original name and create new name
            original_name = field.name
            new_name = original_name.replace(" ", "_")
            original_full_path = f"{prefix}{original_name}"
            new_full_path = f"{prefix}{new_name}"
            name_mapping[original_full_path] = new_full_path

            # Handle different data types
            if isinstance(field.dataType, StructType):
                # Recursively process nested struct
                nested_schema, nested_mapping = transform_schema(field.dataType, f"{new_name}.")
                new_field = StructField(new_name, nested_schema, field.nullable)
                name_mapping.update(nested_mapping)
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                # Handle array of structs
                nested_schema, nested_mapping = transform_schema(field.dataType.elementType, f"{new_name}.")
                new_field = StructField(new_name, ArrayType(nested_schema), field.nullable)
                name_mapping.update({k: v for k, v in nested_mapping.items()})
            elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
                # Handle map with struct values
                nested_schema, nested_mapping = transform_schema(field.dataType.valueType, f"{new_name}.")
                new_field = StructField(new_name, MapType(field.dataType.keyType, nested_schema), field.nullable)
                name_mapping.update({k: v for k, v in nested_mapping.items()})
            else:
                # Simple field
                new_field = StructField(new_name, field.dataType, field.nullable)

            new_fields.append(new_field)

        return StructType(new_fields), name_mapping

    # Create new schema and get name mapping
    new_schema, name_mapping = transform_schema(df.schema)

    # Build select expressions using original names
    def build_select_expr(schema, prefix="", mapping=None):
        exprs = []
        for field in schema.fields:
            new_name = field.name
            full_new_name = f"{prefix}{new_name}"
            # Find original full path from mapping
            original_full_name = next((k for k, v in mapping.items() if v == full_new_name), full_new_name)

            if isinstance(field.dataType, StructType):
                # Handle nested struct
                nested_exprs = build_select_expr(field.dataType, f"{new_name}.", mapping)
                # Use original nested path for selection
                original_struct_name = original_full_name
                exprs.append(struct(nested_exprs).alias(new_name))
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                # Handle array of structs
                nested_exprs = build_select_expr(field.dataType.elementType, "", mapping)
                exprs.append(transform(col(f"`{original_full_name}`"), lambda x: struct(nested_exprs)).alias(new_name))
            else:
                # Simple field
                exprs.append(col(f"`{original_full_name}`").alias(new_name))

        return exprs

    select_expr = build_select_expr(new_schema, mapping=name_mapping)

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
