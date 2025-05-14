from pyspark.sql.types import StructType, StructField, ArrayType, MapType, StringType, IntegerType


def replace_schema_fields_spaces_with_underscores(schema):
    """
    Recursively rename fields in a PySpark schema, replacing spaces with underscores.
    Handles nested StructType, ArrayType, and MapType.
    """

    def rename_field(field):
        # Create new field with renamed name
        new_name = field.name.replace(" ", "_")

        # Handle different data types
        if isinstance(field.dataType, StructType):
            # Recursively process nested struct
            new_data_type = replace_schema_fields_spaces_with_underscores(field.dataType)
            return StructField(new_name, new_data_type, field.nullable, field.metadata)

        elif isinstance(field.dataType, ArrayType):
            # Handle array type
            if isinstance(field.dataType.elementType, StructType):
                # Array of structs - recurse into struct
                new_element_type = replace_schema_fields_spaces_with_underscores(field.dataType.elementType)
                new_data_type = ArrayType(new_element_type, field.dataType.containsNull)
            else:
                # Simple array type
                new_data_type = field.dataType
            return StructField(new_name, new_data_type, field.nullable, field.metadata)

        elif isinstance(field.dataType, MapType):
            # Handle map type
            key_type = field.dataType.keyType
            value_type = field.dataType.valueType

            # Recurse if value type is struct
            if isinstance(value_type, StructType):
                new_value_type = replace_schema_fields_spaces_with_underscores(value_type)
            else:
                new_value_type = value_type

            new_data_type = MapType(key_type, new_value_type, field.dataType.valueContainsNull)
            return StructField(new_name, new_data_type, field.nullable, field.metadata)

        else:
            # Simple data type
            return StructField(new_name, field.dataType, field.nullable, field.metadata)

    # Process all fields in the schema
    if isinstance(schema, StructType):
        new_fields = [rename_field(field) for field in schema.fields]
        return StructType(new_fields)
    else:
        raise ValueError("Input must be a StructType")


# Example usage
def test_rename_schema():
    # Define a complex schema with spaces in names
    complex_schema = StructType([
        StructField("first name", StringType(), True),
        StructField("personal info", StructType([
            StructField("home address", StringType(), True),
            StructField("phone number", StringType(), True),
            StructField("emergency contact", StructType([
                StructField("contact name", StringType(), True),
                StructField("contact phone", StringType(), True)
            ]), True)
        ]), True),
        StructField("test scores", ArrayType(
            StructType([
                StructField("subject name", StringType(), True),
                StructField("score value", IntegerType(), True)
            ])
        ), True),
        StructField("metadata map", MapType(
            StringType(),
            StructType([
                StructField("tag name", StringType(), True),
                StructField("tag value", StringType(), True)
            ])
        ), True)
    ])

    # Rename schema fields
    new_schema = replace_schema_fields_spaces_with_underscores(complex_schema)

    # Print results
    def print_schema(schema, indent=0):
        for field in schema.fields:
            print("  " * indent + f"Field: {field.name}, Type: {field.dataType}")
            if isinstance(field.dataType, StructType):
                print_schema(field.dataType, indent + 1)
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                print("  " * (indent + 1) + "Array element:")
                print_schema(field.dataType.elementType, indent + 2)
            elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
                print("  " * (indent + 1) + "Map value:")
                print_schema(field.dataType.valueType, indent + 2)

    print("Original schema:")
    print_schema(complex_schema)
    print("\nRenamed schema:")
    print_schema(new_schema)


# Run test
test_rename_schema()
