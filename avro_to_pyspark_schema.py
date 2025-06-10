from typing import Union, Dict, List
from avro.schema import Schema, RecordSchema, ArraySchema, UnionSchema, PrimitiveSchema, MapSchema, Field
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, StringType, IntegerType, \
    LongType, DoubleType, FloatType, BooleanType, TimestampType, DateType, BinaryType

def avro_to_pyspark_schema(avro_schema: Union[Schema, str, Dict]) -> StructType:
    """
    Convert an Avro schema to a PySpark schema.
    
    Args:
        avro_schema: Avro schema (can be RecordSchema object, JSON string, or dict)
        
    Returns:
        StructType: Equivalent PySpark schema
    """
    # Parse schema if it's a string or dict
    if isinstance(avro_schema, (str, dict)):
        import avro.schema
        avro_schema = avro.schema.parse(avro_schema) if isinstance(avro_schema, str) else \
                     avro.schema.make_avsc_object(avro_schema)
    
    def _convert_field(field: Field) -> StructField:
        """Convert a single Avro field to a PySpark StructField."""
        field_type = field.type  # Use dot notation instead of subscript
        nullable = field.has_default  # Avro fields with defaults are typically nullable
        
        # Handle different schema types
        if isinstance(field_type, RecordSchema):
            return StructField(field.name, _convert_record(field_type), nullable)
        elif isinstance(field_type, ArraySchema):
            return StructField(field.name, ArrayType(_convert_type(field_type.items), nullable), nullable)
        elif isinstance(field_type, MapSchema):
            return StructField(field.name, MapType(StringType(), _convert_type(field_type.values), nullable), nullable)
        elif isinstance(field_type, UnionSchema):
            # Handle nullable types in union (e.g., ["null", "string"])
            non_null_types = [t for t in field_type.schemas if t.type != 'null']
            spark_type = _convert_type(non_null_types[0]) if non_null_types else StringType()
            return StructField(field.name, spark_type, nullable)
        else:
            return StructField(field.name, _convert_type(field_type), nullable)
    
    def _convert_type(schema: Schema) -> Union[StructType, ArrayType, MapType, StringType, IntegerType, 
                                             LongType, DoubleType, FloatType, BooleanType, 
                                             TimestampType, DateType, BinaryType]:
        """Convert an Avro schema type to a PySpark type."""
        if isinstance(schema, RecordSchema):
            return _convert_record(schema)
        elif isinstance(schema, ArraySchema):
            return ArrayType(_convert_type(schema.items))
        elif isinstance(schema, MapSchema):
            return MapType(StringType(), _convert_type(schema.values))
        elif isinstance(schema, PrimitiveSchema):
            type_mapping = {
                'string': StringType(),
                'int': IntegerType(),
                'long': LongType(),
                'double': DoubleType(),
                'float': FloatType(),
                'boolean': BooleanType(),
                'bytes': BinaryType(),
                'date': DateType(),
                'timestamp-millis': TimestampType(),
                'timestamp-micros': TimestampType()
            }
            return type_mapping.get(schema.type, StringType())  # Default to StringType for unknown types
        elif isinstance(schema, UnionSchema):
            non_null_types = [t for t in schema.schemas if t.type != 'null']
            return _convert_type(non_null_types[0]) if non_null_types else StringType()
        else:
            return StringType()  # Fallback for unhandled types
    
    def _convert_record(record_schema: RecordSchema) -> StructType:
        """Convert an Avro RecordSchema to a PySpark StructType."""
        fields = [_convert_field(field) for field in record_schema.fields]
        return StructType(fields)
    
    if not isinstance(avro_schema, RecordSchema):
        raise ValueError("Avro schema must be a RecordSchema")
    
    return _convert_record(avro_schema)
