import typing as t
import json

from marshmallow import Schema, fields
from marshmallow import ValidationError

from api import config


class TitanicRequestSchema(Schema):
	Age = fields.Float(allow_none=True)
	Cabin = fields.String(allow_none=True)
	Embarked = fields.String(allow_none=True)
	Fare = fields.Float()
	Name = fields.String()
	Parch = fields.Integer()
	PassengerId = fields.Integer()
	Pclass = fields.Integer()
	Sex = fields.String()
	SibSp = fields.Integer()
	Ticket = fields.String()


def validate_inputs(input_data):
	"""Check prediction inputs against schema."""

	#instantiation
	schema = TitanicRequestSchema(many = True)

	errors = ""
	#if errors return none
	try:
		schema.loads(input_data)
	except ValidationError as exc:
		errors = exc.messages
	if errors:
		validated_input = None
	else:
		validated_input = input_data		


	return validated_input, errors



