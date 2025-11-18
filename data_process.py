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

def _parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing dataset name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='WebQSP', help='dataset')
    return parser.parse_args()

def nested_sexpr_to_func_list(split_sexpr, id_now: int = 0):
    """Convert a nested S-expression to agent function call sequence.

    Recursively transforms logical form operators (JOIN, AND, ARG, etc.) into
    executable agent functions with proper expression chaining.

    Args:
        split_sexpr: Nested list representation of S-expression.
        id_now: Current expression ID for variable naming (default: '').

    Returns:
        tuple: (agent_sexpr, function_list)
            - agent_sexpr: List of intermediate expression strings
            - function_list: List of agent function call strings
    """
    # Define function call templates for each operator type
    START = "expression{} = START('{}')"
    JOIN = "expression{} = JOIN({}, {})"
    AND = "expression{} = AND({}, {})"
    ARG = "expression{} = ARG('{}', {}, {})"
    CMP = "expression{} = CMP('{}', {}, {})"
    TC = "expression{} = TC({}, {}, {})"
    COUNT = "expression{} = COUNT({})"
    STOP = "expression = STOP(expression)"

    # Track intermediate expressions and function calls
    agent_sexpr = []
    function_list = []

    # Handle JOIN operation: navigate from entity set via a relation
    if split_sexpr[0]=='JOIN':
        # Handle multi-part string literals (e.g., quoted strings with spaces)
        if len(split_sexpr) > 3 and split_sexpr[2][0]=='"':
            split_sexpr = ['JOIN', split_sexpr[1], ' '.join(split_sexpr[2:])]

        # Convert reverse relation notation: ["R", "relation"] -> "(R relation)"
        if split_sexpr[1][0] == 'R':
            split_sexpr[1] = f'(R {split_sexpr[1][1]})'

        # Process second argument (entity or nested expression)
        if isinstance(split_sexpr[2], list):
            # Recursively process nested expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[2], id_now)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            # Extract expression ID from last function call for reference
            nested_expression_id = nested_func[-1].split(" ", 1)[0]
            split_sexpr_form = nested_sexpr[-1]  # S-expression form for agent_sexpr
        else:
            # Handle entity IDs (m.*, g.*) or special types (date, integer)
            if split_sexpr[2].startswith('m.') or split_sexpr[2].startswith('g.'):
                agent_sexpr.append(f'{split_sexpr[2]}')
                function_list.append(START.format(id_now, split_sexpr[2]))
            else:
                # Special entity types like dates or integers
                agent_sexpr.append(f'{split_sexpr[2]}')
                function_list.append(START.format(id_now, split_sexpr[2]))
            nested_expression_id = f"expression{id_now}"
            split_sexpr_form = split_sexpr[2]

        # Build final JOIN expression
        lhs = split_sexpr[1]
        if not lhs.startswith("expression"):
            lhs = f"'{lhs}'"
        agent_sexpr.append(f'(JOIN {split_sexpr[1]} {split_sexpr_form})')
        function_list.append(JOIN.format(id_now, lhs, nested_expression_id))
        return agent_sexpr, function_list

    # Handle AND operation: intersection of two entity sets
    elif split_sexpr[0]=='AND':
        # Process second argument (base entity set)
        if isinstance(split_sexpr[2], list):
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[2], id_now)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            base_expr_form = nested_sexpr[-1]  # S-expression form
        else:
            agent_sexpr.append(f'{split_sexpr[2]}')
            function_list.append(START.format(id_now, split_sexpr[2]))
            base_expr_form = split_sexpr[2]

        # Process first argument (constraint expression)
        if isinstance(split_sexpr[1], list):
            # Nested expression: create new ID for constraint branch
            next_id = id_now + 1
            constraint_sexpr, constraint_func = nested_sexpr_to_func_list(split_sexpr[1], next_id)
            # Add constraint intermediate expressions
            agent_sexpr += constraint_sexpr
            for func in constraint_func:
                function_list.append(func)
            constraint_expr_form = constraint_sexpr[-1]  # S-expression form
        else:
            # Simple type constraint
            next_id = id_now + 1
            agent_sexpr.append(f'{split_sexpr[1]} {base_expr_form}')
            function_list.append(START.format(next_id, split_sexpr[1]))
            constraint_expr_form = split_sexpr[1]

        # Build final AND expression (intersection)
        agent_sexpr.append(f'(AND {constraint_expr_form} {base_expr_form})')
        function_list.append(AND.format(id_now, f"expression{next_id}", f"expression{id_now}"))
        return agent_sexpr, function_list

    # Handle ARGMAX/ARGMIN: find entity with max/min value of a property
    elif split_sexpr[0] in ['ARGMAX', 'ARGMIN']:
        next_id = id_now + 1

        # Process first argument (entity set to search)
        if isinstance(split_sexpr[1], list):
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[1], next_id)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            entity_set_form = nested_sexpr[-1]  # S-expression form
        else:
            # Simple type-based entity set
            agent_sexpr.append(f'{split_sexpr[1]}')
            function_list.append(START.format(next_id, split_sexpr[1]))
            entity_set_form = split_sexpr[1]

        # Process second argument (comparison property/relation)
        if split_sexpr[2][0] == "R":
            # Convert reverse relation notation
            relation_form = f'(R {split_sexpr[2][1]})'
            relation_ref = f"'{relation_form}'"
        elif isinstance(split_sexpr[2], list):
            # Nested relation expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[2], id_now)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            relation_form = nested_sexpr[-1]  # S-expression form
            relation_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            # Simple single-hop relation (most common case)
            relation_form = split_sexpr[2]
            relation_ref = f"'{relation_form}'"

        # Build final ARG expression
        agent_sexpr.append(f'({split_sexpr[0]} {entity_set_form} {relation_form})')
        op = split_sexpr[0]
        function_list.append(ARG.format(id_now, op, f"expression{next_id}", relation_ref))
        return agent_sexpr, function_list

    # Handle comparison operators: le (<=), lt (<), ge (>=), gt (>)
    elif split_sexpr[0] in ['le', 'lt', 'ge', 'gt']:
        # Process first argument (property/relation to compare)
        if split_sexpr[1][0] == "R":
            # Convert reverse relation notation
            relation_form = f'(R {split_sexpr[1][1]})'
            relation_ref = f"'{relation_form}'"
        elif isinstance(split_sexpr[1], list):
            # Nested relation expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[1], id_now=id_now+1)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            relation_form = nested_sexpr[-1]  # S-expression form
            relation_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            # Simple property name
            relation_form = split_sexpr[1]
            relation_ref = f"'{relation_form}'"

        # Process second argument (comparison value)
        value_form = split_sexpr[2]
        agent_sexpr.append(f'{value_form}')
        function_list.append(START.format(id_now, value_form))

        # Build final comparison expression
        agent_sexpr.append(f'({split_sexpr[0]} {relation_form} {value_form})')
        function_list.append(CMP.format(id_now, split_sexpr[0], relation_ref, f'expression{id_now}'))
        return agent_sexpr, function_list

    # Handle TC (Type Constraint): filter entities by type with direction
    elif split_sexpr[0]=='TC':  # (TC expression relation time)
        # Process first argument (entity set to filter)
        if isinstance(split_sexpr[1], list):
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[1], id_now=id_now+1)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            entity_set_form = nested_sexpr[-1]  # S-expression form
            entity_set_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            entity_set_form = split_sexpr[1]
            agent_sexpr.append(f'{entity_set_form}')
            function_list.append(START.format(id_now, entity_set_form))
            entity_set_ref = f"expression{id_now}"

        # Process second argument (type or relation)
        if split_sexpr[2][0] == "R":
            # Convert reverse relation notation
            type_form = f'(R {split_sexpr[2][1]})'
            type_ref = f"'{type_form}'"
        elif isinstance(split_sexpr[2], list):
            # Nested type expression
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[2], id_now=id_now+2)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            type_form = nested_sexpr[-1]  # S-expression form
            type_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            # Simple type name
            type_form = split_sexpr[2]
            type_ref = f"'{type_form}'"

        # Build final TC expression with direction (split_sexpr[3])
        time_form = split_sexpr[3]
        time_ref = f"'{time_form}'"
        agent_sexpr.append(f'(TC {entity_set_form} {type_form} {time_form})')
        function_list.append(TC.format(id_now, entity_set_ref, type_ref, time_ref))
        return agent_sexpr, function_list

    # Handle COUNT: count number of entities in a set
    elif split_sexpr[0]=='COUNT':
        # Process argument (entity set to count)
        if isinstance(split_sexpr[1], list):
            nested_sexpr, nested_func = nested_sexpr_to_func_list(split_sexpr[1], id_now)
            agent_sexpr += nested_sexpr
            function_list += nested_func
            entity_set_form = nested_sexpr[-1]  # S-expression form
            entity_set_ref = nested_func[-1].split(" ", 1)[0]  # expression ID
        else:
            # Simple expression reference
            entity_set_form = split_sexpr[1]
            entity_set_ref = split_sexpr[1]

        # Build final COUNT expression
        agent_sexpr.append(f'(COUNT {entity_set_form})')
        function_list.append(COUNT.format(id_now, entity_set_ref))
        return agent_sexpr, function_list

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
    if dataset == 'WebQSP':
        from utils.parsing.parse_sparql_webqsp import augment_with_s_expr_webqsp
        dataset_with_sexpr = augment_with_s_expr_webqsp(split, True)
    elif dataset == 'GrailQA':
        from utils.parsing.parse_sparql_grailqa import augment_with_s_expr_grailqa
        dataset_with_sexpr = augment_with_s_expr_grailqa(split, True)
    elif dataset == 'GraphQ':
        from utils.parsing.parse_sparql_graphq import augment_with_s_expr_graphq
        dataset_with_sexpr = augment_with_s_expr_graphq(split, True)

    processed_examples = []

    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Processing {dataset}_{split}'):
        processed_example = {}

        if not example['SExpr_execute_right']:
            continue

        # Extract fields based on dataset format
        if dataset == 'WebQSP':
            # WebQSP has multiple parses - select the shortest valid one
            parses = example['Parses']
            shortest_parse_idx = 0
            shortest_sparql_len = 9999
            for i in range(len(parses)):
                if 'SExpr_execute_right' in parses[i] and parses[i]['SExpr_execute_right']:
                    if len(parses[i]['Sparql']) < shortest_sparql_len:
                        shortest_parse_idx = i
                        shortest_sparql_len = len(parses[i]['Sparql'])

            qid = example['QuestionId']
            question = example['ProcessedQuestion']
            sexpr = parses[shortest_parse_idx]['SExpr']
            sparql = parses[shortest_parse_idx]['Sparql']
            answer = [x['AnswerArgument'] for x in parses[shortest_parse_idx]['Answers']]
        elif dataset == 'GrailQA' or dataset == 'GraphQ':
            # GrailQA and GraphQ have single parse per example
            qid = example['qid']
            question = example['question']
            sexpr = example['s_expression']
            sparql = example['sparql_query']
            answer = example["answer"]

        # Validate S-expression by executing it and comparing with expected answer
        try:
            # Import appropriate SPARQL converter based on dataset
            if dataset == 'WebQSP':
                from utils.executor.logic_form_util import lisp_to_sparql
            elif dataset=='GrailQA' or dataset=='GraphQ':
                from utils.executor.logic_form_util_grailqa import lisp_to_sparql

            # Convert S-expression to SPARQL and execute
            sparql_query = lisp_to_sparql(sexpr)
            query_results = execute_query_with_odbc(sparql_query)

            # Clean Freebase URIs from results
            query_results = [str(res).replace("http://rdf.freebase.com/ns/", '') for res in query_results]

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
        agent_sexpr, function_calls = nested_sexpr_to_func_list(nested_sexpr)
        function_calls.append('expression = STOP(expression)')  # Add termination function

        # Reconstruct S-expression from functions to verify correctness
        reconstructed_sexpr = functions_to_expression(function_calls, 'expression')

        # Validate conversion consistency
        if agent_sexpr[-1] != sexpr:
            pass  # Intermediate check
        if reconstructed_sexpr != sexpr:
            # Skip if reconstruction doesn't match original
            print(sexpr)
            continue

        # Build processed example with all required fields
        processed_example['ID'] = qid
        processed_example['question'] = question
        processed_example['answer'] = answer
        processed_example['sparql'] = sparql
        processed_example['sexpr'] = sexpr
        processed_example['function_list'] = function_calls

        # Add difficulty level for GrailQA test set
        if dataset == 'GrailQA' and split == 'test':
            processed_example['level'] = example["level"]

        # Skip examples with empty answers
        if len(answer) == 0:
            continue

        processed_examples.append(processed_example)

    # Save processed data to output file
    output_dir = f'dataset/{dataset}/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/{dataset}_{split}.json'

    print(f'Processed {len(processed_examples)} valid examples')
    print(f'Writing processed data to {output_file}...')
    dump_json(processed_examples, output_file, indent=4)
    print('Writing finished')

if __name__ == '__main__':
    args = _parse_args()
    # Process both train and test splits
    merge_all_data_for_logical_form_generation(dataset=args.dataset, split="train")
    merge_all_data_for_logical_form_generation(dataset=args.dataset, split="test")
