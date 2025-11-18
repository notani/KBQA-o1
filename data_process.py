"""Data processing module for converting KBQA datasets to agent function format.

This module processes question-answering datasets (WebQSP, GrailQA, GraphQ) by:
1. Parsing logical forms (S-expressions)
2. Converting them to agent function sequences
3. Validating SPARQL queries against Freebase
4. Generating training data with function call representations
"""

import argparse
from tqdm import tqdm
import os
from utils.components.utils import dump_json
from utils.executor.sparql_executor import execute_query_with_odbc
from utils.executor.logic_form_util import lisp_to_nested_expression
from utils.database import functions_to_expression

# Function call templates for S-expression operators
FUNCTION_TEMPLATES = {
    "START": "expression{expr_id} = START('{entity_or_value}')",
    "JOIN": "expression{expr_id} = JOIN({rel}, {exp})",
    "AND": "expression{expr_id} = AND({exp1}, {exp2})",
    "ARG": "expression{expr_id} = ARG('{mode}', {exp}, {rel})",
    "CMP": "expression{expr_id} = CMP('{mode}', {relation}, {exp})",
    "TC": "expression{expr_id} = TC({exp}, {rel}, {time})",
    "COUNT": "expression{expr_id} = COUNT({exp})",
}

# Operators grouped by handling type
COMPARISON_OPS = {"le", "lt", "ge", "gt"}
ARG_OPS = {"ARGMAX", "ARGMIN"}
FREEBASE_ENTITY_PREFIXES = ("m.", "g.")
FREEBASE_URI_PREFIX = "http://rdf.freebase.com/ns/"


def _parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing dataset name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WebQSP", help="dataset")
    return parser.parse_args()


def _get_relation_reference(relation):
    """Extract relation form and reference, handling reverse relations.

    Args:
        relation: Relation value (string or list with reverse notation)

    Returns:
        tuple: (relation_form, relation_ref)
    """
    # Check for reverse relation notation: ["R", "relation"] or starting with "R"
    if isinstance(relation, list) and relation and relation[0] == "R":
        relation_form = f"(R {relation[1]})"
        return relation_form, f"'{relation_form}'"
    elif isinstance(relation, str) and relation and relation[0] == "R":
        relation_form = f"(R {relation[1]})"
        return relation_form, f"'{relation_form}'"
    else:
        # Simple relation name
        return relation, f"'{relation}'"


def _process_entity_value(value, expr_id, expr_trace, func_list):
    """Process entity ID or special value (date, integer).

    Args:
        value: Entity ID or value string
        expr_id: Current expression ID
        expr_trace: List to append intermediate expression forms
        func_list: List to append generated function calls

    Returns:
        tuple: (entity_form, entity_ref_id)
    """
    expr_trace.append(f"{value}")
    func_list.append(
        FUNCTION_TEMPLATES["START"].format(expr_id=expr_id, entity_or_value=value)
    )
    return value, f"expression{expr_id}"


def _process_nested_or_entity(sexpr_arg, expr_id, expr_trace, func_list):
    """Process argument that could be nested expression or simple entity.

    Args:
        sexpr_arg: Argument to process (list for nested, string for entity)
        expr_id: Current expression ID
        expr_trace: List to append intermediate expression forms
        func_list: List to append generated function calls

    Returns:
        tuple: (form_for_sexpr_trace, ref_for_func_list)
    """
    if isinstance(sexpr_arg, list):
        # Recursively process nested expression
        nested_sexpr, nested_func = nested_sexpr_to_func_list(sexpr_arg, expr_id)
        expr_trace.extend(nested_sexpr)
        func_list.extend(nested_func)
        expr_ref = nested_func[-1].split(" ", 1)[0]  # Extract expression ID
        return nested_sexpr[-1], expr_ref
    else:
        # Simple entity or value
        form, ref = _process_entity_value(sexpr_arg, expr_id, expr_trace, func_list)
        return form, ref


def nested_sexpr_to_func_list(sexpr_node, expr_id: int = 0):
    """Convert a nested S-expression to agent function call sequence.

    Recursively transforms logical form operators (JOIN, AND, ARG, etc.) into
    executable agent functions with proper expression chaining.

    Args:
        sexpr_node: Nested list representation of S-expression.
        expr_id: Current expression ID for variable naming (default: 0).

    Returns:
        tuple: (expr_trace, func_list)
            - expr_trace: List of intermediate expression strings
            - func_list: List of agent function call strings
    """
    # Track intermediate expressions and function calls
    expr_trace = []
    func_list = []

    # Handle JOIN operation: navigate from entity set via a relation
    if sexpr_node[0] == "JOIN":
        # Handle multi-part string literals (e.g., quoted strings with spaces)
        if len(sexpr_node) > 3 and sexpr_node[2][0] == '"':
            sexpr_node = ["JOIN", sexpr_node[1], " ".join(sexpr_node[2:])]

        # Convert reverse relation notation: ["R", "relation"] -> "(R relation)"
        if sexpr_node[1] and sexpr_node[1][0] == "R":
            sexpr_node[1] = f"(R {sexpr_node[1][1]})"

        # Process second argument (entity or nested expression)
        sexpr_node_form, nested_expression_id = _process_nested_or_entity(
            sexpr_node[2], expr_id, expr_trace, func_list
        )

        # Build final JOIN expression
        rel = sexpr_node[1]
        if not rel.startswith("expression"):
            rel = f"'{rel}'"
        expr_trace.append(f"(JOIN {sexpr_node[1]} {sexpr_node_form})")
        func_list.append(
            FUNCTION_TEMPLATES["JOIN"].format(
                expr_id=expr_id, rel=rel, exp=nested_expression_id
            )
        )
        return expr_trace, func_list

    # Handle AND operation: intersection of two entity sets
    elif sexpr_node[0] == "AND":
        # Process second argument (base entity set)
        base_expr_form, _ = _process_nested_or_entity(
            sexpr_node[2], expr_id, expr_trace, func_list
        )

        # Process first argument (constraint expression)
        next_id = expr_id + 1
        if isinstance(sexpr_node[1], list):
            # Nested expression: create new ID for constraint branch
            constraint_sexpr, constraint_func = nested_sexpr_to_func_list(
                sexpr_node[1], next_id
            )
            expr_trace.extend(constraint_sexpr)
            func_list.extend(constraint_func)
            constraint_expr_form = constraint_sexpr[-1]  # S-expression form
        else:
            # Simple type constraint
            expr_trace.append(f"{sexpr_node[1]} {base_expr_form}")
            func_list.append(
                FUNCTION_TEMPLATES["START"].format(
                    expr_id=next_id, entity_or_value=sexpr_node[1]
                )
            )
            constraint_expr_form = sexpr_node[1]

        # Build final AND expression (intersection)
        expr_trace.append(f"(AND {constraint_expr_form} {base_expr_form})")
        func_list.append(
            FUNCTION_TEMPLATES["AND"].format(
                expr_id=expr_id,
                exp1=f"expression{next_id}",
                exp2=f"expression{expr_id}",
            )
        )
        return expr_trace, func_list

        # Handle ARGMAX/ARGMIN: find entity with max/min value of a property
    elif sexpr_node[0] in ARG_OPS:
        next_id = expr_id + 1

        # Process first argument (entity set to search)
        entity_set_form, _ = _process_nested_or_entity(
            sexpr_node[1], next_id, expr_trace, func_list
        )

        # Process second argument (comparison property/relation)
        if isinstance(sexpr_node[2], list):
            # Nested relation expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(
                sexpr_node[2], expr_id
            )
            expr_trace.extend(nested_sexpr)
            func_list.extend(nested_func)
            relation_form = nested_sexpr[-1]  # S-expression form
            relation_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            relation_form, relation_ref = _get_relation_reference(sexpr_node[2])

        # Build final ARG expression
        expr_trace.append(f"({sexpr_node[0]} {entity_set_form} {relation_form})")
        func_list.append(
            FUNCTION_TEMPLATES["ARG"].format(
                expr_id=expr_id,
                mode=sexpr_node[0],
                exp=f"expression{next_id}",
                rel=relation_ref,
            )
        )
        return expr_trace, func_list

    # Handle comparison operators: le (<=), lt (<), ge (>=), gt (>)
    elif sexpr_node[0] in COMPARISON_OPS:
        # Process first argument (property/relation to compare)
        if isinstance(sexpr_node[1], list):
            # Nested relation expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(
                sexpr_node[1], expr_id=expr_id + 1
            )
            expr_trace.extend(nested_sexpr)
            func_list.extend(nested_func)
            relation_form = nested_sexpr[-1]  # S-expression form
            relation_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            relation_form, relation_ref = _get_relation_reference(sexpr_node[1])

        # Process second argument (comparison value)
        value_form = sexpr_node[2]
        expr_trace.append(f"{value_form}")
        func_list.append(
            FUNCTION_TEMPLATES["START"].format(
                expr_id=expr_id, entity_or_value=value_form
            )
        )

        # Build final comparison expression
        expr_trace.append(f"({sexpr_node[0]} {relation_form} {value_form})")
        func_list.append(
            FUNCTION_TEMPLATES["CMP"].format(
                expr_id=expr_id,
                mode=sexpr_node[0],
                relation=relation_ref,
                exp=f"expression{expr_id}",
            )
        )
        return expr_trace, func_list

        # Handle TC (Type Constraint): filter entities by type with direction
    elif sexpr_node[0] == "TC":  # (TC expression relation time)
        # Process first argument (entity set to filter)
        entity_set_form, entity_set_ref = _process_nested_or_entity(
            sexpr_node[1], expr_id + 1, expr_trace, func_list
        )
        if not entity_set_ref.startswith("expression"):
            expr_trace.append(f"{entity_set_form}")
            func_list.append(
                FUNCTION_TEMPLATES["START"].format(
                    expr_id=expr_id, entity_or_value=entity_set_form
                )
            )
            entity_set_ref = f"expression{expr_id}"

        # Process second argument (type or relation)
        if isinstance(sexpr_node[2], list):
            # Nested type expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(
                sexpr_node[2], expr_id=expr_id + 2
            )
            expr_trace.extend(nested_sexpr)
            func_list.extend(nested_func)
            type_form = nested_sexpr[-1]  # S-expression form
            type_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            type_form, type_ref = _get_relation_reference(sexpr_node[2])

        # Build final TC expression with direction (sexpr_node[3])
        time_form = sexpr_node[3]
        time_ref = f"'{time_form}'"
        expr_trace.append(f"(TC {entity_set_form} {type_form} {time_form})")
        func_list.append(
            FUNCTION_TEMPLATES["TC"].format(
                expr_id=expr_id, exp=entity_set_ref, rel=type_ref, time=time_ref
            )
        )
        return expr_trace, func_list

    # Handle COUNT: count number of entities in a set
    elif sexpr_node[0] == "COUNT":
        # Process argument (entity set to count)
        entity_set_form, entity_set_ref = _process_nested_or_entity(
            sexpr_node[1], expr_id, expr_trace, func_list
        )

        # Build final COUNT expression
        expr_trace.append(f"(COUNT {entity_set_form})")
        func_list.append(
            FUNCTION_TEMPLATES["COUNT"].format(expr_id=expr_id, exp=entity_set_ref)
        )
        return expr_trace, func_list

    # Unknown operator - should not reach here in valid expressions
    else:
        pass


def merge_all_data_for_logical_form_generation(dataset, split):
    """Process and merge KBQA dataset into agent function format.

    Loads a dataset split, converts S-expressions to agent functions,
    validates SPARQL queries, and saves the processed data.

    Args:
        dataset: Dataset name ('WebQSP', 'GrailQA', or 'GraphQ').
        split: Dataset split ('train' or 'test').
    """
    # Load dataset with S-expressions based on dataset type
    if dataset == "WebQSP":
        from utils.parsing.parse_sparql_webqsp import augment_with_s_expr_webqsp

        dataset_with_sexpr = augment_with_s_expr_webqsp(split, True)
    elif dataset == "GrailQA":
        from utils.parsing.parse_sparql_grailqa import augment_with_s_expr_grailqa

        dataset_with_sexpr = augment_with_s_expr_grailqa(split, True)
    elif dataset == "GraphQ":
        from utils.parsing.parse_sparql_graphq import augment_with_s_expr_graphq

        dataset_with_sexpr = augment_with_s_expr_graphq(split, True)

    processed_examples = []

    for example in tqdm(
        dataset_with_sexpr,
        total=len(dataset_with_sexpr),
        desc=f"Processing {dataset}_{split}",
    ):
        processed_example = {}

        if not example["SExpr_execute_right"]:
            continue

        # Extract fields based on dataset format
        if dataset == "WebQSP":
            # WebQSP has multiple parses - select the shortest valid one
            parses = example["Parses"]
            shortest_parse_idx = 0
            shortest_sparql_len = 9999
            for i in range(len(parses)):
                if (
                    "SExpr_execute_right" in parses[i]
                    and parses[i]["SExpr_execute_right"]
                ):
                    if len(parses[i]["Sparql"]) < shortest_sparql_len:
                        shortest_parse_idx = i
                        shortest_sparql_len = len(parses[i]["Sparql"])

            qid = example["QuestionId"]
            question = example["ProcessedQuestion"]
            sexpr = parses[shortest_parse_idx]["SExpr"]
            sparql = parses[shortest_parse_idx]["Sparql"]
            answer = [
                x["AnswerArgument"] for x in parses[shortest_parse_idx]["Answers"]
            ]
        elif dataset == "GrailQA" or dataset == "GraphQ":
            # GrailQA and GraphQ have single parse per example
            qid = example["qid"]
            question = example["question"]
            sexpr = example["s_expression"]
            sparql = example["sparql_query"]
            answer = example["answer"]

        # Validate S-expression by executing it and comparing with expected answer
        try:
            # Import appropriate SPARQL converter based on dataset
            if dataset == "WebQSP":
                from utils.executor.logic_form_util import lisp_to_sparql
            elif dataset == "GrailQA" or dataset == "GraphQ":
                from utils.executor.logic_form_util_grailqa import lisp_to_sparql

            # Convert S-expression to SPARQL and execute
            sparql_query = lisp_to_sparql(sexpr)
            query_results = execute_query_with_odbc(sparql_query)

            # Clean Freebase URIs from results
            query_results = [
                str(res).replace(FREEBASE_URI_PREFIX, "") for res in query_results
            ]

            # Skip if query results don't match expected answers
            if set(query_results) != set(answer):
                print(sexpr)
                continue
        except KeyboardInterrupt:
            break
        except Exception:
            # Skip examples that fail to execute
            print(sexpr)
            continue

        # Use the validated SPARQL query
        sparql = sparql_query

        # Convert S-expression to agent function sequence
        nested_sexpr = lisp_to_nested_expression(sexpr)
        expr_trace, function_calls = nested_sexpr_to_func_list(nested_sexpr)
        last_expression_id = function_calls[-1].split(" ", 1)[0]
        function_calls.append(
            f"expression = STOP({last_expression_id})"
        )  # Add termination function

        # Reconstruct S-expression from functions to verify correctness
        reconstructed_sexpr = functions_to_expression(function_calls, "expression")

        # Validate conversion consistency
        if expr_trace[-1] != sexpr:
            pass  # Intermediate check
        if reconstructed_sexpr != sexpr:
            # Skip if reconstruction doesn't match original
            print(sexpr)
            continue

        # Build processed example with all required fields
        processed_example["ID"] = qid
        processed_example["question"] = question
        processed_example["answer"] = answer
        processed_example["sparql"] = sparql
        processed_example["sexpr"] = sexpr
        processed_example["function_list"] = function_calls

        # Add difficulty level for GrailQA test set
        if dataset == "GrailQA" and split == "test":
            processed_example["level"] = example["level"]

        # Skip examples with empty answers
        if len(answer) == 0:
            continue

        processed_examples.append(processed_example)

    # Save processed data to output file
    output_dir = f"dataset/{dataset}/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/{dataset}_{split}.json"

    print(f"Processed {len(processed_examples)} valid examples")
    print(f"Writing processed data to {output_file}...")
    dump_json(processed_examples, output_file, indent=4)
    print("Writing finished")


if __name__ == "__main__":
    args = _parse_args()
    # Process both train and test splits
    merge_all_data_for_logical_form_generation(dataset=args.dataset, split="train")
    merge_all_data_for_logical_form_generation(dataset=args.dataset, split="test")
