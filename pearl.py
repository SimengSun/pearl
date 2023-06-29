import csv
import pdb
import json
from utils import *
import argparse
from tqdm import tqdm

openai_setapi()

def mine_actions(args):
    # get converted questions
    fname = "./data/processed/quality_train_q.csv"
    questions = load_questions(fname)
    qfilter = load_question_filter()
    questions = {qid: questions[qid] for qid in questions if qid not in qfilter['train_long']}
    print(f"number of questions: {len(questions)}")
    prompt = load_prompt(f"./prompt_bank/mine_actions.txt")

    # mine actions
    with open(args.output_file, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "question"])
        writer.writeheader()
        ret = []
        for qid in tqdm(questions):
            item = questions[qid]
            this_ret = {"qid": qid}
            this_question = item['question']
            this_prompt = prompt.replace("{{question}}", this_question)
            response = get_response(this_prompt,
                                    model=model_name,
                                    max_tokens=512,
                                    frequency_penalty=0.0,
                                    temperature=0.0,
                                    top_p=0.0,
                                    stop=["<|im_end|>", "\n\n\n", "---"])
            this_ret['question'] = response
            ret.append(this_ret)
            writer.writerow(this_ret)

    # aggregate actions
    all_new_actions = []
    for item in ret:
        this_actions = item['question']
        try:
            new_actions = this_actions[:this_actions.index("My sequence of actions:")]
            new_actions = new_actions[new_actions.index("\n")+1:]
            new_actions = [x.strip() for x in new_actions.split("\n") if "none" not in x.lower() and len(x.strip()) > 0]
            all_new_actions.extend(new_actions)
        except:
            continue
    all_new_actions = set(all_new_actions)
    print(f"number of actions: {len(all_new_actions)}")
    
    merged_actions = {}
    for action in all_new_actions:
        action_name = action.split(":")[0].strip()
        definition = action.split(":")[1].strip()
        merged_actions[action_name] = definition

    out_str = ''
    for action in merged_actions:
        out_str += f"{action.lstrip('-').strip()}\t#{merged_actions[action]}\n"
    
    with open(args.output_file, "w") as f:
        f.write(out_str)

def simplify_actions(args):
    input_actions = load_text(args.input_file).split('\n')
    simplify_prompt = load_prompt(f"./prompt_bank/simplify_actions.txt")
    shard_size = args.shard_size
    num_shards = len(input_actions) // shard_size + 1
    output = ''
    for i in range(num_shards):
        this_shard = input_actions[i*shard_size : min((i+1)*shard_size, len(input_actions))]
        this_total_actions = '\n'.join(this_shard)
        this_prompt = simplify_prompt.replace("{action_list}", this_total_actions)
        response = get_response(this_prompt,
                            model=model_name, 
                            frequency_penalty=0, 
                            temperature=0.0, 
                            top_p=0.0,
                            stop=["\n\n\n"],
                            max_tokens=2048)
        output += response.strip() + "\n"
    
    with open(args.output_file, "w") as f:
        f.write(output)

def load_actions():
    global all_actions
    all_actions = {}

    with open(f"./output/mined_actions_simplified_example.txt", "r") as f:
        for line in f:
            try:
                lsp = line.split("#")
                action_type = lsp[0][:lsp[0].index("(")]
                action_args = lsp[0][lsp[0].index("(")+1:lsp[0].index(")")].split(",")
                action_def = lsp[1].strip()
                if action_type in all_actions:
                    print(f"Warning: {action_type} already exists")
                all_actions[action_type] = {"args": [x.strip() for x in action_args], "action_def": action_def}
            except:
                pdb.set_trace()

    print(len(all_actions))

def get_option_str(question):
    options = ''
    for idx in range(1, 5):
        option = question[f'option_{idx}']
        options += f'{option_map[idx]}: {option}\n'
    return options

def load_article(fname):
    with open(fname, "r") as f:
        data = f.readlines()
        data = [json.loads(x) for x in data]
        # get rid of excessive newlines
        data = {x['article_id']: process_article(x['article'], chunk_size=-1) for x in data}
        return data

def load_quality_data(this_split, this_type):
    qfilter = load_question_filter()
    if this_split == "dev":
        articles = load_article("./data/raw/QuALITY.v1.0.1.htmlstripped.dev")
        questions = load_questions("./data/processed/quality_dev_q.csv")
        if this_type == "ctx_eval_long":
            long_qids = qfilter['dev_long']
        elif this_type == "ctx_eval_short":
            long_qids = qfilter['dev_short']
        else:
            raise ValueError(f"Unknown example type {this_type}")
        questions = {qid:questions[qid] for qid in questions if qid in long_qids}
    elif this_split == "train":
        articles = load_article("./data/raw/QuALITY.v1.0.1.htmlstripped.train")
        questions = load_questions("./data/processed/quality_train_q.csv")
        if this_type == "ctx_eval_long":
            long_qids = qfilter['train_long']
        else:
            raise ValueError(f"Unknown example type {this_type}")
        questions = {qid:questions[qid] for qid in questions if qid in long_qids}
    elif this_split == "train-demo":
        articles = load_article("./data/raw/QuALITY.v1.0.1.htmlstripped.train")
        questions = load_questions("./data/processed/quality_train_q.csv")
        qids = qfilter['train_demo']
        questions = {qid:questions[qid] for qid in questions if qid in qids}
    else:
        raise ValueError(f"Unknown split {this_split}")
    print(f"Loaded {len(articles)} articles and {len(questions)} questions")
    return articles, questions

def load_csv(fname):
    with open(fname, "r") as f:
        reader = csv.DictReader(f)
        ret = {}
        for row in reader:
            ret[row["qid"]] = row
    return ret

class Action(object):
    def __init__(self, question, entire_plan, action_type, detailed_action, action_def=None, current_action=None):

        global all_actions
        self.this_action = action_type
        self.args = all_actions[action_type]["args"]
        self.action_def = all_actions[action_type]["action_def"] if action_def is None else action_def
        original_action = f"{action_type}({','.join(self.args)})"
        self.entire_plan = [x for x in entire_plan.split("\n\n") if len(x.strip()) > 0][1]
        self.question = question
        self.current_action = current_action
        self.this_prompt = ""
        if "CTX" in original_action:
            self.this_prompt += "{{CTX}}\n---\n\nPlease read the above text first, and then follow the instructions below.\n\n"

        self.this_prompt += f"[Instruction]\nAction:\n\n{original_action} : {self.action_def}\n\nthis_args\n---\n\n[Answer]\n(list or paragraph(s), please be thorough)\n({detailed_action})\n"   

    def execute(self, *args):
        try:
            assert len(args) == len(self.args)

            if 'current_action' in self.this_prompt:
                self.this_prompt = self.this_prompt.replace('current_action', self.current_action)

            if "{{CTX}}" in self.this_prompt:
                self.this_prompt = self.this_prompt.replace("{{CTX}}", args[0])
                args = args[1:]
                self.args = self.args[1:]

            if 'current_action' not in self.this_prompt:
                args_str = ""
                for i in range(len(args)):
                    args_str += f"{self.args[i]} = {args[i]}\n"
                self.this_prompt = self.this_prompt.replace("this_args", args_str)
            
            response = get_response(self.this_prompt,
                                    model=model_name, 
                                    frequency_penalty=0, 
                                    temperature=0.0, 
                                    top_p=0.0,
                                    stop=["\n\n\n"],
                                    max_tokens=512)
        except Exception as e:
            print(e)
            print(self.this_prompt)
            print(args)
            print(self.args)
        return response
    
    def __call__(self, *args):
        if self.this_action != "CONCAT":
            return self.execute(*args)
        else:
            return "\n".join(args)

def generate_plan(question,  
                invalid_plan=None, 
                error_message=None, 
                all_error_messages=None, 
                debug=False, 
                plan_prompt=None,
                plan_prompt_invalid=None):

    action_list = load_text("./output/mined_actions_simplified_example.txt")

    if invalid_plan is None:
        plan_generation_prompt = load_prompt(plan_prompt)
        plan_generation_prompt = plan_generation_prompt.format(action_list=action_list, question=question)
    else:
        plan_generation_prompt = load_prompt(plan_prompt_invalid)
        plan_generation_prompt = plan_generation_prompt.format(action_list=action_list, question=question, invalid_plan=invalid_plan, error_message=error_message, all_error_messages='\n\t' + '\n\t'.join(all_error_messages))
    if debug:
        print(plan_generation_prompt)
    response = get_response(plan_generation_prompt,
                        model=model_name, 
                        frequency_penalty=0, 
                        temperature=0.0, 
                        top_p=0.0,
                        stop=["\n\n\n"],
                        max_tokens=256)
    if debug:
        print(response)

    plan = response
    return plan

def parse_plan(plan):
    """
    Input: plan of the format:
            New actions:
            - new_action_1(args) : one sentence of explanation

            output_1 = action_1(args for action_1) : explanation
            output_2 = action_2(args for action_2) : explanation
            ...

    Output (if valid plan):
            is_valid
            actions: a list of actions, each item is a map of the format:
                    {"action": action_name, 
                    "args": [arg1, arg2, ...], 
                    "output_var": output_name,
                    "detailed_action": detailed_action_string,
                    'action_def': action_def if it is a new action}
            output_map: a map of the format:
                    {"output_1": None, "output_2": None, ...}
    
    Output (if invalid plan):
            is_valid
            error message
            invalid plan
    """
    global all_actions
    # separate new actions from plans
    plan_sp = [x for x in plan.split("\n\n") if len(x.strip()) > 0]
    if len(plan_sp) != 2:
        error_message = "Invalid plan: Need to have two parts (new actions and plan) separated by a blank line."
        return False, error_message, "\n\n".join(plan_sp)

    new_actions = plan_sp[0]
    plan = plan_sp[1]

    # parse new actions
    try:
        new_actions = [x.replace("- ", "").strip() for x in new_actions.split("\n")[1:] if len(x.strip()) > 0]
        this_new_actions = {}
        for line in new_actions:
            if "none" in line.lower():
                break
            lsp = line.split(":")
            action_type = lsp[0][:lsp[0].index("(")]
            action_args = lsp[0][lsp[0].index("(")+1:lsp[0].index(")")].split(",")
            action_def = lsp[1].strip()
            all_actions[action_type] = {"args": [x.strip() for x in action_args], "action_def": action_def}
    except:
        error_message = "Invalid plan: new actions format is incorrect."
        return False, error_message, '\n\n'.join(plan_sp)
    
    # parse plan
    plan = [x.strip() for x in plan.split("\n") if len(x.strip()) > 0]
    output_map = {}
    actions = []
    for row in plan:
        try:
            row = row[row.index(".")+1:].strip()
        except:
            error_message = f"Invalid plan: no number index and '.' found in action \n\t{row}."
            return False, error_message, "\n".join(plan)

        try:
            row = row.split("=")
        except:
            error_message = "Invalid plan: no '=' found in one of the actions"
            return False, error_message, "\n".join(plan)

        try:
            output = row[0].strip()
            output_map[output] = None
            action_and_args = row[1][:row[1].index(":")].strip()
            action_and_args = action_and_args.split("(")
            action_definitions = row[1][row[1].index(":")+1:].strip()
            action = action_and_args[0].strip()
            args = [x.lstrip().rstrip() for x in action_and_args[1][:-1].split(",")]
            action_map = {"action": action, "args": args, "output_var": output, "detailed_action": action_definitions}
            actions.append(action_map)
        except:
            error_message = "Error parsing plan. Plan format is incorrect. Please check the plan format."
            return False, error_message, "\n".join(plan)
        
        for action in actions:
            if action["action"] not in all_actions and action["action"] not in this_new_actions:
                error_message = f"Error parsing action {action['action']}. Unknown action."
                return False, error_message, "\n".join(plan)
            
            if action["action"] in this_new_actions:
                action["action_def"] = this_new_actions[action["action"]]["action_def"]
                defined_args = this_new_actions[action["action"]]["args"]
            else:
                defined_args = all_actions[action["action"]]["args"]

            this_args = action["args"]

            if action["action"] != "CONCAT":
                if len(defined_args) != len(this_args):
                    error_message = f"Error parsing action {action['action']}. Number of arguments is incorrect"
                    return False, error_message, "\n".join(plan)

            if action["output_var"] in this_args:
                error_message = f"Error parsing action {action['action']}. Output variable is used as an argument"
                return False, error_message, "\n".join(plan)

            for arg in action["args"]:
                if arg == "CTX":
                    continue
                if arg not in output_map and "\"" not in arg:
                    error_message = f"Error parsing action {action['action']}. Argument {arg} is not defined."
                    return False, error_message, "\n".join(plan)
            
    return True, actions, output_map

def execute_plan(actions, plan, question, output_map, article, debug=False):
    """
        Input:
            actions: a list of actions, each item is a map of the format:
                    {"action": action_name, 
                    "args": [arg1, arg2, ...], 
                    "output_var": output_name
                    "detailed_action": detailed_action_string,
                    "action_def": action_def if it is a new action}
            plan: the plan in string format
            output_map: a map of the format 
                    {"output_1": None, "output_2": None, ...} which stores 
                    the value of each output variable
            article: the article
            debug: whether to print debug information
        Output:
            end_response: concatenation of last step output and intermediate output if it is not fed as input to other actions
    """
    all_args = []
    max_len = 8192
    reslen = 512
    for action in actions:
        action_name = action["action"]
        args = action["args"]
        current_action = f'{action_name}({", ".join(args)})'
        all_args.extend(args)
        args = [x if x == "CTX" or "\"" in x else output_map[x] for x in args]
        args = [article if x == "CTX" else x for x in args]
        if "action_def" in action:
            action_func = Action(question, plan, action_name, action["detailed_action"], action_def=action["action_def"], current_action=current_action) 
        else:
            action_func = Action(question, plan, action_name, action["detailed_action"], current_action=current_action)
        action_func_prompt = action_func.this_prompt
        if sum([len(enc.encode(x)) for x in args]) + len(enc.encode(action_func_prompt)) + reslen + 1 > max_len:
            truncated_idx = sum([len(enc.encode(x)) for x in args]) + len(enc.encode(action_func_prompt)) + reslen + 1 - max_len
            args[0] = enc.decode(enc.encode(args[0])[:-truncated_idx]) + "..."
        try:
            output = action_func(*args)
        except:
            print("Error executing action")
            print(action)
            print(args)
            print(output_map)
            return None
        
        output_map[action["output_var"]] = output + "\n"
        if debug:
            print(action)
            print(output)
            print("="*23)
    
    # TODO: new
    end_response = ""
    for action in actions:
        if action["output_var"] not in all_args:
            end_response += output_map[action["output_var"]] + "\n\n"
    return end_response

def _pearl(args, article, qid, question, options, invalid_plan=None, all_error_messages=[]):
    """
        execute pearl for individual example
    """
    error_message = None
    retry_cnt = 0
    all_error_messages = all_error_messages
    while True:
        # generate plan, if plan is invalid, ask the model to correct+refine the plan
        plan_str = generate_plan(question["question"],  
                                invalid_plan=invalid_plan, 
                                error_message=error_message, 
                                all_error_messages=all_error_messages, 
                                debug=args.debug, 
                                plan_prompt=args.prompt_plan_file,
                                plan_prompt_invalid=args.prompt_plan_invalid_file,)
        # parse plan
        is_valid, out_1, out_2 = parse_plan(plan_str)
        if is_valid:
            plan = out_1
            output_map = out_2
            break
        else:
            error_message = out_1
            invalid_plan = plan_str
            retry_cnt += 1
            print(error_message)
            all_error_messages.append(error_message)
            if retry_cnt > 7:
                break

        if retry_cnt > 7:
            print(f"Error: {qid}")
            print("Need to fallback to baseline open-answer")
            continue

    if args.debug:
        print(f"Plan: {plan}")

    # execute plan 
    response = execute_plan(plan, plan_str, question["question"], output_map, article, debug=args.debug)
    # map open answer to choice
    this_map_prompt = map_prompt.format(open_answer=response, question=question["question"], options=options)

    map_response = get_response(this_map_prompt,
                                model=model_name,
                                frequency_penalty=0,
                                temperature=0.0,
                                top_p=0.0,
                                stop=["\n\n\n"],
                                max_tokens=4)[0]

    if args.debug:
        print(f"Answer: {response}")
        print(f"Map prompt: {this_map_prompt}")
        print(f"Map answer: {map_response}")
        print("="*20)
    
    res_dict = {"qid": qid, 
                "plan": plan_str, 
                "open-answer": response, 
                "map-answer": map_response, 
                "gold": option_map[int(question['gold_label'])]}
    return res_dict, output_map

def refine(args):
    '''
        Refine the demonstration examples that are incoporated into the plan formulation stage
        The demonstration examples should not be any examples the model is evaluated on
    '''
    global debug
    debug = args.debug

    articles, questions = load_quality_data("train-demo", None)
    with open(args.output_file + f".train_demo.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "plan", "open-answer", "map-answer", "gold"])
        writer.writeheader()

        for qi, qid in enumerate(tqdm(questions)):
            try:
                question = questions[qid]
                options = get_option_str(question)
                article = articles[question["article_id"]]

                retry_cnt = 0
                all_error_messages = []
                first_incorrect_plan = None
                invalid_plan = None
                while True:
                    res_dict, _ = _pearl(args, article, qid, question, options, invalid_plan=invalid_plan, all_error_messages=all_error_messages)
                    if res_dict["map-answer"] == res_dict["gold"]:
                        break
                    else:
                        invalid_plan = res_dict["plan"]
                        if first_incorrect_plan is None:
                            first_incorrect_plan = invalid_plan
                        retry_cnt += 1
                        all_error_messages.append(f"{invalid_plan.lstrip().rstrip()}\n\nError: Incorrect plan. Could not map to correct answer. Please rethink the plan strategy.\n\n")
                        if retry_cnt > 3:
                            break
                if retry_cnt > 3:
                    continue
                writer.writerow(res_dict)

            except Exception as e:
                print(f"Error: {qid}")
                print(e)
                continue

    # print output
    with open(args.output_file + f".train_demo.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
            question = questions[qid]['question'].strip()
            plan = row["plan"]

            print(f"Question: \"{question}\"\n\nAnswer:\n{plan}\n---\n")

def baseline_mcq(args, this_split="dev", this_type="ctx_eval_long"):
    # load prompt
    prompt = "Article\n\n{article}End of Article\n\nQuestion:{question}\n{options}\n\nRead the article and answer the question by selecting the best option. Only one of the options is correct.\n\nAnswer (select from A, B, C, D):\n"
    
    articles, questions = load_quality_data(this_split, this_type)

    # for each question, generate open-ended answer with options and write to file
    # write to file
    with open(args.output_file + f".{this_split}.{this_type}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer", "gold"])
        writer.writeheader()
        total_cnt = 0
        crrc_cnt = 0
        for qid in tqdm(questions):
            try:
                question = questions[qid]
                options = get_option_str(question)
                article = articles[question["article_id"]]
                this_prompt = prompt.format(article=article, question=question["question"], options=options)
                
                len_this_prompt = len(enc.encode(this_prompt))
                if len_this_prompt + max_output_len > 8192:
                    if args.debug:
                        print(f"Exceed length limit: {len_this_prompt}")
                    article = enc.decode(enc.encode(article)[:-(len_this_prompt + max_output_len - 8192 + 1)])
                    this_prompt = prompt.format(article=article, question=question["question"], options=options)
                
                response = get_response(this_prompt,
                                        model=model_name, 
                                        frequency_penalty=0, 
                                        temperature=0.0, 
                                        top_p=0.0,
                                        stop=["\n\n\n"],
                                        max_tokens=5)[0]
                os.system("sleep 5s")  
                if args.debug:
                    print(f"Prompt: {this_prompt}")
                    print(f"Answer: {response}")
                    print("="*20)
                writer.writerow({"qid": qid, "answer": response, 'gold': option_map[int(question['gold_label'])]})
                total_cnt += 1
                crrc_cnt += 1 if response == option_map[int(question['gold_label'])] else 0
                print(f"Accuracy: {crrc_cnt / total_cnt}")

            except Exception as e:
                print(e)
                continue
        print(f"Accuracy: {crrc_cnt / total_cnt}") 

def baseline_gqa(args, this_split="dev", this_type="ctx_eval_long"):
    # load prompt
    this_prompt_template = load_prompt(f"./prompt_bank/freeform_ans.txt")
    
    articles, questions = load_quality_data(this_split, this_type)

    # for each question, generate open-ended answer with options and write to file
    # write to file
    with open(args.output_file + f".{this_split}.{this_type}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "open-answer", "map-answer", "gold"])
        writer.writeheader()
        total_cnt = 0
        crrc_cnt = 0

        for qid in tqdm(questions):
            try:
                question = questions[qid]
                options = get_option_str(question)
                article = articles[question["article_id"]]
                this_prompt = this_prompt_template.format(article=article, question=question["question"])
                len_this_prompt = len(enc.encode(this_prompt))
                if len_this_prompt + max_output_len > 8192:
                    if args.debug:
                        print(f"Exceed length limit: {len_this_prompt}")
                    article = enc.decode(enc.encode(article)[:-(len_this_prompt + max_output_len - 8192 + 1)])
                    this_prompt = this_prompt_template.format(article=article, question=question["question"])
                
                response = get_response(this_prompt,
                                        model=model_name, 
                                        frequency_penalty=0, 
                                        temperature=0.0, 
                                        top_p=0.0,
                                        stop=["\n\n\n"],
                                        max_tokens=max_output_len)
                os.system("sleep 5s") # change this if there aren't any excessive rate limit errors

                this_map_prompt = map_prompt.format(open_answer=response, question=question["question"], options=options)
                
                map_response = get_response(this_map_prompt,
                                            model=model_name,
                                            frequency_penalty=0,
                                            temperature=0.0,
                                            top_p=0.0,
                                            stop=["\n\n\n"],
                                            max_tokens=4)[0]
                if args.debug:
                    print(f"Prompt: {this_prompt}")
                    print(f"Answer: {response}")
                    print(f"Map prompt: {this_map_prompt}")
                    print(f"Map answer: {map_response}")
                    print("="*20)

                writer.writerow({"qid": qid, "open-answer": response, "map-answer": map_response, 'gold': option_map[int(question['gold_label'])]})
                total_cnt += 1
                crrc_cnt += 1 if map_response == option_map[int(question['gold_label'])] else 0
                print(f"Accuracy: {crrc_cnt/total_cnt}")
            except  Exception as e:
                print(f"Error: {qid}")
                print(e)
                continue

def pearl(args, this_split="dev", this_type="ctx_eval_long"):

    global debug 
    debug = args.debug

    articles, questions = load_quality_data(this_split, this_type)

    all_output_map = {}
    with open(args.output_file + f".{this_split}.{this_type}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "plan", "open-answer", "map-answer", "gold"])
        writer.writeheader()
        total_cnt = 0
        crrc_cnt = 0
        
        for qi, qid in enumerate(tqdm(questions)):
            try:
                question = questions[qid]
                options = get_option_str(question)
                article = articles[question["article_id"]]

                res_dict, output_map = _pearl(args, article, qid, question, options)
                writer.writerow(res_dict)
                
                all_output_map[qid] = output_map

                total_cnt += 1
                if res_dict["map-answer"] == res_dict["gold"]:
                    crrc_cnt += 1
                
                print(f"Accuracy: {crrc_cnt/total_cnt}")

            except Exception as e:
                print(f"Error: {qid}")
                print(e)
                continue
    
    # save output_map to pickle
    output_map_file = args.output_file + f".{this_split}.{this_type}.output_map.pkl"
    with open(output_map_file, "wb") as f:
        pickle.dump(all_output_map, f, protocol=pickle.HIGHEST_PROTOCOL)
                   
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["mine_actions", "simplify_actions", "refine", "baseline_mcq", "baseline_gqa", "pearl"])
    parser.add_argument("--prompt-plan-file", type=str, default="")
    parser.add_argument("--prompt-plan-invalid-file", type=str, default="")
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--shard-size", type=int, default=80, help="number of actions per shard during action simplification")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.stage == "mine_actions":
        mine_actions(args)

    elif args.stage == "simplify_actions":
        simplify_actions(args)

    elif args.stage == "refine":
        load_actions()
        refine(args)

    elif args.stage == "baseline_mcq":
        baseline_mcq(args, this_split="dev", this_type="ctx_eval_long")
        baseline_mcq(args, this_split="train", this_type="ctx_eval_long")
        baseline_mcq(args, this_split="dev", this_type="ctx_eval_short")

    elif args.stage == "baseline_gqa":
        baseline_gqa(args, this_split="dev", this_type="ctx_eval_long")
        baseline_gqa(args, this_split="train", this_type="ctx_eval_long")
        baseline_gqa(args, this_split="dev", this_type="ctx_eval_short")

    elif args.stage == "pearl":
        load_actions()
        pearl(args, this_split="dev", this_type="ctx_eval_long")
        pearl(args, this_split="train", this_type="ctx_eval_long")
        pearl(args, this_split="dev", this_type="ctx_eval_short")
        
    else:
        raise ValueError("Unknown stage")

if __name__ == "__main__":
    main()