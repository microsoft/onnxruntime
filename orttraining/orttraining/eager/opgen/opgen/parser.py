# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Optional, Tuple, Union

from opgen.ast import *  # noqa: F403
from opgen.lexer import *  # noqa: F403


class UnexpectedTokenError(RuntimeError):
    def __init__(self, expected: TokenKind, actual: Token):  # noqa: F405
        self.expected = expected
        self.actual = actual
        super().__init__(f"unexpected token {actual}; expected {expected}")


class ExpectedSyntaxError(RuntimeError):
    def __init__(self, expected: str, actual: Token = None):  # noqa: F405
        super().__init__(f"expected {expected}; actual {actual}")


class ParserBase:
    _peek_queue: List[Token]  # noqa: F405

    def __init__(self, lexer: Union[Lexer, Reader]):  # noqa: F405
        self._own_lexer = False
        if isinstance(lexer, Reader):  # noqa: F405
            self._own_lexer = True
            lexer = Lexer(lexer)  # noqa: F405
        elif not isinstance(lexer, Lexer):  # noqa: F405
            raise TypeError("lexer must be a Lexer or Reader")
        self._lexer = lexer
        self._peek_queue = []

    def __enter__(self):
        if self._own_lexer:
            self._lexer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._own_lexer:
            self._lexer.__exit__(exc_type, exc_val, exc_tb)

    def set_source_location(self, origin: SourceLocation):  # noqa: F405
        self._lexer.set_source_location(origin)

    def _peek_token(
        self, kinds: Union[TokenKind, List[TokenKind]] = None, value: str = None, look_ahead: int = 1  # noqa: F405
    ) -> Optional[Token]:  # noqa: F405
        if look_ahead < 1:
            raise IndexError("look_ahead must be at least 1")
        if look_ahead >= len(self._peek_queue):
            for _ in range(look_ahead - len(self._peek_queue)):
                self._peek_queue = [self._lexer.lex(), *self._peek_queue]
        peek = self._peek_queue[-look_ahead]
        if not kinds:
            return peek
        if not isinstance(kinds, list):
            kinds = [kinds]
        for kind in kinds:
            if peek.kind == kind:
                if value:
                    return peek if peek.value == value else None
                return peek
        return None

    def _read_token(self) -> Token:  # noqa: F405
        return self._peek_queue.pop() if self._peek_queue else self._lexer.lex()

    def _expect_token(self, kind: TokenKind) -> Token:  # noqa: F405
        token = self._read_token()
        if token.kind != kind:
            raise UnexpectedTokenError(kind, token)
        return token

    def _parse_list(
        self,
        open_token_kind: TokenKind,  # noqa: F405
        separator_token_kind: TokenKind,  # noqa: F405
        close_token_kind: TokenKind,  # noqa: F405
        member_parser: callable,
    ) -> SyntaxList:  # noqa: F405
        syntax_list = SyntaxList()  # noqa: F405
        if open_token_kind:
            syntax_list.open_token = self._expect_token(open_token_kind)
        while True:
            if close_token_kind and self._peek_token(close_token_kind):
                break
            member = member_parser()
            if not self._peek_token(separator_token_kind):
                syntax_list.append(member, None)
                break
            syntax_list.append(member, self._read_token())
        if close_token_kind:
            syntax_list.close_token = self._expect_token(close_token_kind)
        return syntax_list

    def parse_translation_unit(self) -> TranslationUnitDecl:  # noqa: F405
        decls = []
        while not self._peek_token(TokenKind.EOF):  # noqa: F405
            decls.append(self.parse_function())
        return TranslationUnitDecl(decls)  # noqa: F405

    def parse_function_parameter_default_value_expression(self) -> Expression:  # noqa: F405
        return self.parse_expression()

    def parse_function_parameter(self) -> ParameterDecl:  # noqa: F405
        parameter_type = self.parse_type()

        if not self._peek_token(TokenKind.IDENTIFIER):  # noqa: F405
            return ParameterDecl(parameter_type)  # noqa: F405

        parameter_name = self._read_token()

        if not self._peek_token(TokenKind.EQUALS):  # noqa: F405
            return ParameterDecl(parameter_type, parameter_name)  # noqa: F405

        return ParameterDecl(  # noqa: F405
            parameter_type, parameter_name, self._read_token(), self.parse_function_parameter_default_value_expression()
        )

    def parse_function_parameters(self) -> SyntaxList:  # noqa: F405
        return self._parse_list(
            TokenKind.OPEN_PAREN, TokenKind.COMMA, TokenKind.CLOSE_PAREN, self.parse_function_parameter  # noqa: F405
        )

    def parse_function(self) -> FunctionDecl:  # noqa: F405
        raise NotImplementedError()

    def parse_expression(self) -> Expression:  # noqa: F405
        raise NotImplementedError()

    def parse_type(self) -> Type:  # noqa: F405
        raise NotImplementedError()


class CPPParser(ParserBase):
    def parse_function(self) -> FunctionDecl:  # noqa: F405
        return_type = self.parse_type()
        return FunctionDecl(  # noqa: F405
            self._expect_token(TokenKind.IDENTIFIER),  # noqa: F405
            self.parse_function_parameters(),
            return_type,
            semicolon=self._expect_token(TokenKind.SEMICOLON),  # noqa: F405
        )

    def parse_expression(self) -> Expression:  # noqa: F405
        if (
            self._peek_token(TokenKind.IDENTIFIER)  # noqa: F405
            or self._peek_token(TokenKind.NUMBER)  # noqa: F405
            or self._peek_token(TokenKind.STRING)  # noqa: F405
        ):
            return LiteralExpression(self._read_token())  # noqa: F405
        else:
            raise UnexpectedTokenError("expression", self._peek_token())

    def parse_type(self) -> Type:  # noqa: F405
        if self._peek_token(TokenKind.IDENTIFIER, "const"):  # noqa: F405
            parsed_type = ConstType(self._read_token(), self.parse_type())  # noqa: F405
        elif self._peek_token([TokenKind.IDENTIFIER, TokenKind.DOUBLECOLON]):  # noqa: F405
            identifiers = []
            while True:
                token = self._peek_token([TokenKind.IDENTIFIER, TokenKind.DOUBLECOLON])  # noqa: F405
                if not token:
                    break
                identifiers.append(self._read_token())
                if token.has_trailing_trivia(TokenKind.WHITESPACE):  # noqa: F405
                    break
            if self._peek_token(TokenKind.LESS_THAN):  # noqa: F405
                parsed_type = TemplateType(  # noqa: F405
                    identifiers,
                    self._parse_list(
                        TokenKind.LESS_THAN,  # noqa: F405
                        TokenKind.COMMA,  # noqa: F405
                        TokenKind.GREATER_THAN,  # noqa: F405
                        self._parse_template_type_argument,
                    ),
                )
            elif identifiers[-1].value == "TensorOptions":
                parsed_type = TensorOptionsType(identifiers)  # noqa: F405
            else:
                parsed_type = ConcreteType(identifiers)  # noqa: F405
        else:
            raise ExpectedSyntaxError("type", self._peek_token())

        while True:
            if self._peek_token(TokenKind.AND):  # noqa: F405
                parsed_type = ReferenceType(parsed_type, self._read_token())  # noqa: F405
            else:
                return parsed_type

    def _parse_template_type_argument(self) -> Type:  # noqa: F405
        if self._peek_token(TokenKind.NUMBER):  # noqa: F405
            return ExpressionType(self.parse_expression())  # noqa: F405
        return self.parse_type()


class TorchParser(ParserBase):
    def __init__(self, lexer: Union[Lexer, Reader]):  # noqa: F405
        super().__init__(lexer)
        self._next_anonymous_alias_id = 0

    def parse_function(self) -> FunctionDecl:  # noqa: F405
        return FunctionDecl(  # noqa: F405
            self._expect_token(TokenKind.IDENTIFIER),  # noqa: F405
            self.parse_function_parameters(),
            arrow=self._expect_token(TokenKind.ARROW),  # noqa: F405
            return_type=self.parse_type(),
        )

    def parse_expression(self) -> Expression:  # noqa: F405
        if (
            self._peek_token(TokenKind.NUMBER)  # noqa: F405
            or self._peek_token(TokenKind.IDENTIFIER)  # noqa: F405
            or self._peek_token(TokenKind.STRING)  # noqa: F405
        ):
            return LiteralExpression(self._read_token())  # noqa: F405
        elif self._peek_token(TokenKind.OPEN_BRACKET):  # noqa: F405
            return ArrayExpression(  # noqa: F405
                self._parse_list(
                    TokenKind.OPEN_BRACKET,  # noqa: F405
                    TokenKind.COMMA,  # noqa: F405
                    TokenKind.CLOSE_BRACKET,  # noqa: F405
                    self.parse_expression,
                )
            )
        else:
            raise UnexpectedTokenError("expression", self._peek_token())

    def _create_alias_info_type(self, parsed_type: Type, alias_info: AliasInfo) -> AliasInfoType:  # noqa: F405
        if isinstance(parsed_type, ModifiedType):  # noqa: F405
            parsed_type.base_type = AliasInfoType(parsed_type.base_type, alias_info)  # noqa: F405
        else:
            parsed_type = AliasInfoType(parsed_type, alias_info)  # noqa: F405
        return parsed_type

    def parse_type(self) -> Type:  # noqa: F405
        parsed_type, alias_info = self._parse_type_and_alias()
        if not alias_info:
            return parsed_type
        return self._create_alias_info_type(parsed_type, alias_info)

    def _parse_type_and_alias(self) -> Tuple[Type, AliasInfo]:  # noqa: F405
        parsed_type: Type = None  # noqa: F405
        alias_info: AliasInfo = None  # noqa: F405

        if self._peek_token(TokenKind.MUL):  # noqa: F405
            return (KWArgsSentinelType(self._read_token()), None)  # noqa: F405

        if self._peek_token(TokenKind.OPEN_PAREN):  # noqa: F405

            def parse_tuple_element():
                element_type, element_alias_info = self._parse_type_and_alias()
                if element_alias_info:
                    element_type = self._create_alias_info_type(element_type, element_alias_info)

                return TupleMemberType(  # noqa: F405
                    element_type, self._read_token() if self._peek_token(TokenKind.IDENTIFIER) else None  # noqa: F405
                )

            parsed_type = TupleType(  # noqa: F405
                self._parse_list(
                    TokenKind.OPEN_PAREN,  # noqa: F405
                    TokenKind.COMMA,  # noqa: F405
                    TokenKind.CLOSE_PAREN,  # noqa: F405
                    parse_tuple_element,
                )
            )
        elif self._peek_token(TokenKind.IDENTIFIER, "Tensor"):  # noqa: F405
            parsed_type = TensorType(self._read_token())  # noqa: F405
            alias_info = self._parse_torch_alias_info()
        else:
            parsed_type = self._parse_torch_base_type()
            alias_info = self._parse_torch_alias_info()

        while True:
            if self._peek_token(TokenKind.OPEN_BRACKET):  # noqa: F405
                parsed_type = ArrayType(  # noqa: F405
                    parsed_type,
                    self._read_token(),
                    self._read_token() if self._peek_token(TokenKind.NUMBER) else None,  # noqa: F405
                    self._expect_token(TokenKind.CLOSE_BRACKET),  # noqa: F405
                )
            elif self._peek_token(TokenKind.QUESTION_MARK):  # noqa: F405
                parsed_type = OptionalType(parsed_type, self._read_token())  # noqa: F405
            else:
                return (parsed_type, alias_info)

    def _parse_torch_base_type(self) -> Type:  # noqa: F405
        base_type_parsers = {
            "int": IntType,  # noqa: F405
            "float": FloatType,  # noqa: F405
            "bool": BoolType,  # noqa: F405
            "str": StrType,  # noqa: F405
            "Scalar": ScalarType,  # noqa: F405
            "ScalarType": ScalarTypeType,  # noqa: F405
            "Dimname": DimnameType,  # noqa: F405
            "Layout": LayoutType,  # noqa: F405
            "Device": DeviceType,  # noqa: F405
            "Generator": GeneratorType,  # noqa: F405
            "MemoryFormat": MemoryFormatType,  # noqa: F405
            "QScheme": QSchemeType,  # noqa: F405
            "Storage": StorageType,  # noqa: F405
            "ConstQuantizerPtr": ConstQuantizerPtrType,  # noqa: F405
            "Stream": StreamType,  # noqa: F405
            "SymInt": SymIntType,  # noqa: F405
        }
        identifier = self._expect_token(TokenKind.IDENTIFIER)  # noqa: F405
        base_type_parser = base_type_parsers.get(identifier.value)
        if not base_type_parser:
            raise ExpectedSyntaxError("|".join(base_type_parsers.keys()), identifier)
        base_type = base_type_parser(identifier)
        return base_type

    def _parse_torch_alias_info(self) -> AliasInfo:  # noqa: F405
        alias_info = AliasInfo()  # noqa: F405

        def parse_set(alias_set: List[str]):
            while True:
                if self._peek_token(TokenKind.MUL):  # noqa: F405
                    alias_info.tokens.append(self._read_token())
                    alias_set.append("*")
                elif "*" not in alias_set:
                    identifier = self._expect_token(TokenKind.IDENTIFIER)  # noqa: F405
                    alias_info.tokens.append(identifier)
                    alias_set.append(identifier.value)
                else:
                    raise ExpectedSyntaxError("alias wildcard * or alias identifier", self._peek_token())

                if self._peek_token(TokenKind.OR):  # noqa: F405
                    alias_info.tokens.append(self._read_token())
                else:
                    return

        if self._peek_token(TokenKind.OPEN_PAREN):  # noqa: F405
            alias_info.tokens.append(self._read_token())

            parse_set(alias_info.before_set)

            if self._peek_token(TokenKind.EXCLAIMATION_MARK):  # noqa: F405
                alias_info.tokens.append(self._read_token())
                alias_info.is_writable = True

            if self._peek_token(TokenKind.ARROW):  # noqa: F405
                alias_info.tokens.append(self._read_token())
                parse_set(alias_info.after_set)
            else:
                # no '->' so assume before and after are identical
                alias_info.after_set = alias_info.before_set

            alias_info.tokens.append(self._expect_token(TokenKind.CLOSE_PAREN))  # noqa: F405
        elif self._peek_token(TokenKind.EXCLAIMATION_MARK):  # noqa: F405
            alias_info.is_writable = True
            alias_info.before_set.append(str(self._next_anonymous_alias_id))
            self._next_anonymous_alias_id += 1
        else:
            return None

        return alias_info


def cpp_create_from_file(path: str) -> CPPParser:
    return CPPParser(FileReader(path))  # noqa: F405


def cpp_create_from_string(buffer: str) -> CPPParser:
    return CPPParser(StringReader(buffer))  # noqa: F405


def torch_create_from_file(path: str) -> TorchParser:
    return TorchParser(FileReader(path))  # noqa: F405


def torch_create_from_string(buffer: str) -> TorchParser:
    return TorchParser(StringReader(buffer))  # noqa: F405
