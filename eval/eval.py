
import argparse
import os
import sys
import logging
import json

import logging
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Dict
logger = logging.getLogger(__name__)

class Group():
    def __init__(self, id, msg, roles, form, stwr, isNested=False, hasNested=False):
        self.id: str = id
        self.msg: List[int] = msg
        self.roles: Dict[str, List[int]] = roles
        self.form: str = form
        self.stwr: str = stwr
        self.isNested: bool = isNested
        self.hasNested: bool = hasNested

class Document():
    
    def __init__(self, doc_id, annotations):
        self.trigger = ["quote"]
        self.role_set = ["addressee", "cue", "frame", "speaker"]
        self.messages = []
        self.groups = []
        self.doc_id = doc_id  # os.path.splitext(os.path.basename(file_name))[0]
        self.parse_tags(annotations)

    @property
    def id(self):
        return self.doc_id

    def parse_tags(self, annotations):       
        for id, annot in enumerate(annotations):
            msg = self.convert_to_abs(annot["quote"]["tokenIds"])
            self.messages.append(("quote", msg))
            logger.debug("Role_set "+str(self.role_set))
            roles = {}
            for role in self.role_set:
                r = self.convert_to_abs(annot.get(role, {}).get("tokenIds", []))
                roles[role] = r
            self.groups.append(Group(id, msg, roles, annot.get("type", None), annot.get("medium", None), isNested=annot.get("isNested", False)))
    
    def convert_to_abs(self, sent_toks):
        sent_toks.sort()
        return remove_consecutive_duplicates(sent_toks)


class ScoreTracker():
    """ a class that keeps track of the scores for a variable unit """
    
    def __init__(self,fid, roles):
        self.fid=fid
        self.parts = roles
        self.types = ["Direct", "Indirect", "FreeIndirect","IndirectFreeIndirect", "Reported"]
        self.mediums = ["Speech", "Thought", "Writing", "ST", "SW", "TW", "none"]
        self.tp_parts = {r: [] for r in self.parts}
        self.fp_parts = {r: [] for r in self.parts}
        self.fn_parts = {r: [] for r in self.parts}
        self.tp_types = {f: [] for f in self.types}
        self.fp_types = {f: [] for f in self.types}
        self.fn_types = {f: [] for f in self.types}
        self.tp_mediums = {s: [] for s in self.mediums}
        self.fp_mediums = {s: [] for s in self.mediums}
        self.fn_mediums = {s: [] for s in self.mediums}

    def report(self):
        return "".join( [ "\n"+"\t"*8,"TP_msg:\t",str(len(self.tp_parts["quote"])), "\tFP_msg:\t",str(len(self.fp_parts["quote"])), 
        "\tFN_msg:\t",str(len(self.fn_parts["quote"])),"\n"+"\t"*8,"TP_roles:\t",str(sum(len(self.tp_parts[r]) for r in self.parts[1:])),
        "\tFP_roles:\t",str(sum(len(self.fp_parts[r]) for r in self.parts[1:])),"\tFN_roles:\t",str(sum(len(self.fn_parts[r]) for r in self.parts[1:])) ])

    def update(self,other):
        for r in self.parts:
            self.tp_parts[r].extend(other.tp_parts[r])
            self.fp_parts[r].extend(other.fp_parts[r])
            self.fn_parts[r].extend(other.fn_parts[r])
        for f in self.types:
            self.tp_types[f].extend(other.tp_types[f])
            self.fp_types[f].extend(other.fp_types[f])
            self.fn_types[f].extend(other.fn_types[f])
        for s in self.mediums:
            self.tp_mediums[s].extend(other.tp_mediums[s])
            self.fp_mediums[s].extend(other.fp_mediums[s])
            self.fn_mediums[s].extend(other.fn_mediums[s])

    def get_macro_recall(self):
        values = []
        for r in self.parts:
            num = float(len(self.tp_parts[r]))
            denum = num + float(len(self.fn_parts[r]))
            values.append(num / denum if denum > 0 else None)
        return tuple(values)
    
    def get_macro_precision(self):
        values = []
        for r in self.parts:
            num = float(len(self.tp_parts[r]))
            denum = num + float(len(self.fp_parts[r]))
            values.append(num / denum if denum > 0 else None)
        return tuple(values)

    def get_micro_recall_for_type(self, typelabel):
        try:
            if typelabel.lower() == 'quote':
                num =  float(len(self.tp_parts["quote"]))
                denum = num + float(len(self.fn_parts["quote"]))
                return num / denum
            elif typelabel.lower() == 'role':
                num =  float(sum(len(self.tp_parts[x]) for x in self.parts[1:]))
                denum = num + float(sum(len(self.fn_parts[x]) for x in self.parts[1:]))
                return num / denum
            elif typelabel.lower() == "joint":
                num =  float(sum(len(self.tp_parts[x]) for x in self.parts))
                denum = num + float(sum(len(self.fn_parts[x]) for x in self.parts))
                return num / denum
            elif typelabel.lower() == "type":
                num =  float(sum(len(self.tp_types[x]) for x in self.types))
                denum = num + float(sum(len(self.fn_types[x]) for x in self.types))
                return num / denum
            elif typelabel.lower() == "medium":
                num =  float(sum(len(self.tp_mediums[x]) for x in self.mediums))
                denum = num + float(sum(len(self.fn_mediums[x]) for x in self.mediums))
                return num / denum
            else:
                raise ValueError(f"wrong typelabel {typelabel}")
        except ZeroDivisionError:
            return 0.0

    def get_micro_precision_for_type(self, typelabel):

        try:
            if typelabel.lower() == 'quote':
                num = float(len(self.tp_parts["quote"]))
                denum = num + float(len(self.fp_parts["quote"]))
                return num / denum
            elif typelabel.lower() == 'role':
                num = float(sum(len(self.tp_parts[x]) for x in self.parts[1:]))
                denum = num + float(sum(len(self.fp_parts[x]) for x in self.parts[1:]))
                return num / denum
            elif typelabel.lower() == "joint":
                num = float(sum(len(self.tp_parts[x]) for x in self.parts))
                denum = num + float(sum(len(self.fp_parts[x]) for x in self.parts))
                return num / denum
            elif typelabel.lower() == "type":
                num = float(sum(len(self.tp_types[x]) for x in self.types))
                denum = num + float(sum(len(self.fp_types[x]) for x in self.types))
                return num / denum
            elif typelabel.lower() == "medium":
                num = float(sum(len(self.tp_mediums[x]) for x in self.mediums))
                denum = num + float(sum(len(self.fp_mediums[x]) for x in self.mediums))
                return num / denum
            else:
                raise ValueError(f"wrong typelabel {typelabel}")
        except ZeroDivisionError:
            return 0.0

    # we set beta to 1 (same weight for precision and recall => harmonic mean)
    def get_F_beta(self, p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0


class Evaluator():
    def __init__(self, sys_sas, gs_sas):
        self.doc_ids = []
        self.scores = {}
        self.roles = ["quote", "addressee", "cue", "frame", "speaker"]
        self.nested = True
        self.perform_evaluation(sys_sas, gs_sas)
        
    @staticmethod
    def recall(tp, fn):
        try:
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def precision(tp, fp):
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0
    @staticmethod
    def F_beta(p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0
    
    def perform_evaluation(self, sys_sas, gs_sas):
        glob_eval=ScoreTracker("all", self.roles)

        all_group_evals = []

        for doc_id in sorted(list(gs_sas.keys())):
            logger.info("evaluating subtrack 1 on "+doc_id)
            # start an evaluation tracker for the evaluation of this document
            
            doc_eval=ScoreTracker(doc_id, self.roles)
            logger.debug("initialized doc_eval for "+doc_id+" "  +doc_eval.report())
            
            gold_groups = gs_sas[doc_id].groups
            sys_groups = sys_sas[doc_id].groups if doc_id in sys_sas else []
            if not self.nested:
                gold_groups = [g for g in gold_groups if not g.isNested]
                sys_groups = [g for g in sys_groups if not g.isNested]

            group_evaluations = self.evaluate_doc(sys_groups, gold_groups, doc_eval)
            all_group_evals.extend(group_evaluations)

            
            logger.info("Eval for doc "+doc_id)
            logger.info(doc_eval.report())
            doc_p= doc_eval.get_micro_precision_for_type("quote")
            logger.info("doc msg micro precision "+str(doc_p))
            doc_r= doc_eval.get_micro_recall_for_type("quote")
            logger.info("doc msg micro recall "+str(doc_r))
            doc_f1=doc_eval.get_F_beta(doc_p,doc_r)
            
            logger.info("doc msg F1 "+str(doc_f1))
            doc_role_p =doc_eval.get_micro_precision_for_type("role")
            doc_role_r= doc_eval.get_micro_recall_for_type("role")
            doc_role_f1=doc_eval.get_F_beta(doc_role_p,doc_role_r)
            logger.info("doc role Prec "+str(doc_role_p))
            logger.info("doc role Rec "+str(doc_role_r))
            logger.info("doc role F1 "+str(doc_role_f1))

            doc_joint_p =doc_eval.get_micro_precision_for_type("joint")
            doc_joint_r= doc_eval.get_micro_recall_for_type("joint")
            doc_joint_f1=doc_eval.get_F_beta(doc_joint_p,doc_joint_r)

            logger.info("doc joint Prec "+str(doc_joint_p))
            logger.info("doc joint Rec "+str(doc_joint_r))
            logger.info("doc joint F1 "+str(doc_joint_f1))
            glob_eval.update(doc_eval)
            logger.info("")
            logger.info("next document")
            
            self.doc_ids.append(doc_id)
            continue
     
        # micro scores
        logger.info("Global evaluation "+str(glob_eval.report()))
        g_prec_cue = glob_eval.get_micro_precision_for_type("quote")
        logger.info("global prec msg: "+str(g_prec_cue))
        g_rec_cue  = glob_eval.get_micro_recall_for_type("quote")
        logger.info("global recall msg: "+str(g_rec_cue))
        g_f1_cue   = glob_eval.get_F_beta(g_prec_cue,g_rec_cue)
        logger.info("global f1 msg: "+str(g_f1_cue))

        
        g_prec_rol = glob_eval.get_micro_precision_for_type("role")
        logger.info("global prec roles: "+str(g_prec_rol))
        g_rec_rol  = glob_eval.get_micro_recall_for_type("role")
        logger.info("global recall roles: "+str(g_rec_rol))
        g_f1_rol = glob_eval.get_F_beta(g_prec_rol,g_rec_rol)
        logger.info("global f1 roles: "+str(g_f1_rol))

        g_prec_joint = glob_eval.get_micro_precision_for_type("joint")
        logger.info("global prec joint: "+str(g_prec_joint))
        g_rec_joint  = glob_eval.get_micro_recall_for_type("joint")
        logger.info("global recall joint: "+str(g_rec_joint))
        g_f1_joint = glob_eval.get_F_beta(g_prec_joint,g_rec_joint)
        logger.info("global f1 joint: "+str(g_f1_joint))

        form_precision = glob_eval.get_micro_precision_for_type("type")
        form_recall = glob_eval.get_micro_recall_for_type("type")
        form_f1 = glob_eval.get_F_beta(form_precision, form_recall)

        stwr_precision = glob_eval.get_micro_precision_for_type("medium")
        stwr_recall = glob_eval.get_micro_recall_for_type("medium")
        stwr_f1 = glob_eval.get_F_beta(stwr_precision, stwr_recall)

        # macro scores
        g_precision_values = [[] for _ in range(len(glob_eval.parts))]
        g_recall_values = [[] for _ in range(len(glob_eval.parts))]
        for g in all_group_evals:
            for i, p in enumerate(g.get_macro_precision()):
                if p is not None:
                    g_precision_values[i].append(p)
            for i, r in enumerate(g.get_macro_recall()):
                if r is not None:
                    g_recall_values[i].append(r)
        # g_precision_values = [p for p in (g.get_macro_precision() for g in all_group_evals) if p is not None]
        g_macro_precision_joint = sum(p for role in g_precision_values for p in role) / sum(len(role) for role in g_precision_values)
        g_macro_precision_msg = sum(p for p in g_precision_values[0]) / len(g_precision_values[0])
        g_macro_precision_roles = sum(p for role in g_precision_values[1:] for p in role) / sum(len(role) for role in g_precision_values[1:])
        # g_recall_values = [r for r in (g.get_macro_recall() for g in all_group_evals) if r is not None]
        g_macro_recall_joint = sum(r for role in g_recall_values for r in role) / sum(len(role) for role in g_recall_values)
        g_macro_recall_msg = sum(r for r in g_recall_values[0]) / len(g_recall_values[0])
        g_macro_recall_roles = sum(r for role in g_recall_values[1:] for r in role) / sum(len(role) for role in g_recall_values[1:])
        g_macro_f1_joint = glob_eval.get_F_beta(g_macro_precision_joint, g_macro_recall_joint)
        g_macro_f1_msg = glob_eval.get_F_beta(g_macro_precision_msg, g_macro_recall_msg)
        g_macro_f1_roles = glob_eval.get_F_beta(g_macro_precision_roles, g_macro_recall_roles)

        logger.info("Macro precision: %s", g_macro_precision_joint)
        logger.info("Macro recall: %s", g_macro_recall_joint)
        logger.info("Macro F1: %s", g_macro_f1_joint)
        
        self.scores = {
            'prec_msg': g_macro_precision_msg,
            'rec_msg': g_macro_recall_msg,
            'f1_msg': g_macro_f1_msg,
            'prec_roles': g_macro_precision_roles,
            'rec_roles': g_macro_recall_roles,
            'f1_roles': g_macro_f1_roles,
            'prec_joint': g_macro_precision_joint,
            'rec_joint': g_macro_recall_joint,
            'f1_joint': g_macro_f1_joint,
            'prec_form': form_precision,
            'rec_form': form_recall,
            'f1_form': form_f1,
            'prec_stwr': stwr_precision,
            'rec_stwr': stwr_recall,
            'f1_stwr': stwr_f1,
        }    
    
    def evaluate_doc(self, groups_sys: List[Group], groups_gold: List[Group], doc_eval: ScoreTracker):
        doc_evaluations = []

        assigned_sys, assigned_gold = assign_messages(groups_sys, groups_gold)
        if len(assigned_sys) != len(assigned_gold):
            logger.warning("Error in assignment")
        for si, gi in zip(assigned_sys, assigned_gold):
            gs = groups_sys[si]
            gg = groups_gold[gi]
            group_eval = ScoreTracker(f"sys_{gs.id}-gold_{gg.id}", self.roles)
            evaluate_group(gs, gg, group_eval)
            doc_eval.update(group_eval)
            doc_evaluations.append(group_eval)
        
        empty_group = Group(None, [], {r: [] for r in doc_eval.parts}, None, None)

        unmatched_sys = set(range(len(groups_sys))).difference(assigned_sys)
        for si in unmatched_sys:
            gs = groups_sys[si]
            group_eval = ScoreTracker(f"sys_{gs.id}-gold_None", self.roles)
            evaluate_group(gs, empty_group, group_eval)
            doc_eval.update(group_eval)
            doc_evaluations.append(group_eval)

        unmatched_gold = set(range(len(groups_gold))).difference(assigned_gold)
        for gi in unmatched_gold:
            gg = groups_gold[gi]
            group_eval = ScoreTracker(f"sys_None-gold_{gg.id}", self.roles)
            evaluate_group(empty_group, gg, group_eval)
            doc_eval.update(group_eval)
            doc_evaluations.append(group_eval)
        logger.info("Evaluated document %s", doc_eval.fid)
        return doc_evaluations
   
    def print_report(self, file):
        logger.info(self.__class__.__name__)
        logger.debug("Class Evaluate printing report...")
        logger.debug("Evaluate obj prints report")
        self._print_summary(file)

    def _print_summary(self, file):
        file.write("Messages(F1): {}\n".format(self.scores['f1_msg']))
        file.write("Messages(P):  {}\n".format(self.scores['prec_msg']))
        file.write("Messages(R):  {}\n\n".format(self.scores['rec_msg']))

        file.write("Roles(F1): {}\n".format(self.scores['f1_roles']))
        file.write("Roles(P):  {}\n".format(self.scores['prec_roles']))
        file.write("Roles(R):  {}\n\n".format(self.scores['rec_roles']))
        
        file.write("Joint(F1): {}\n".format(self.scores['f1_joint']))
        file.write("Joint(P):  {}\n".format(self.scores['prec_joint']))
        file.write("Joint(R):  {}\n\n".format(self.scores['rec_joint']))

        file.write("Form(F1): {}\n".format(self.scores['f1_form']))
        file.write("Form(P):  {}\n".format(self.scores['prec_form']))
        file.write("Form(R):  {}\n\n".format(self.scores['rec_form']))

        file.write("STWR(F1): {}\n".format(self.scores['f1_stwr']))
        file.write("STWR(P):  {}\n".format(self.scores['prec_stwr']))
        file.write("STWR(R):  {}\n\n".format(self.scores['rec_stwr']))


def remove_consecutive_duplicates(lst):
    # temporary variable to store the last seen element
    last_seen = None
    res = []
    for x in lst:
        if x != last_seen:
            res.append(x)
        last_seen = x
    return res

def compute_overlap(gold, sys):
    onlyG = 0
    onlyS = 0
    both = 0
    g = 0
    s = 0
    while g < len(gold) and s < len(sys):
        if gold[g] == sys[s]:
            both += 1
            g += 1
            s += 1
        elif gold[g] < sys[s]:
            onlyG += 1
            g += 1
        else:
            onlyS += 1
            s += 1
    onlyG += len(gold) - g
    onlyS += len(sys) - s
    if both + onlyS != len(sys):
        logger.warning("compute_overlap error")
    if both + onlyG != len(gold):
        logger.warning("compute_overlap error")
    accuracy =  both / (both + onlyG + onlyS)
    precision = both / (both + onlyS)
    recall = both / (both + onlyG)
    f_1 = 0.0 if (precision + recall == 0.0) else 2 * precision * recall / (precision + recall)
    return f_1

def score_spans(gold, sys, tp: List, fp: List, fn: List):
    g = 0
    s = 0
    while g < len(gold) and s < len(sys):
        if gold[g] == sys[s]:
            tp.append(gold[g])
            g += 1
            s += 1
        elif gold[g] < sys[s]:
            fn.append(gold[g])
            g += 1
        else:
            fp.append(sys[s])
            s += 1
    fn.extend(gold[g:])
    fp.extend(sys[s:])

def assign_messages(group_sys: List[Group], group_gold: List[Group]):
    scores = np.zeros((len(group_gold), len(group_sys)))
    for i, gg in enumerate(group_gold):
        for j, gs in enumerate(group_sys):
            overlap = compute_overlap(gg.msg, gs.msg)
            # use same form/STWR as tie-breaker
            form_bonus = 0.01 if overlap > 0.0 and gg.form == gs.form else 0.0
            stwr_bonus = 0.001 if overlap > 0.0 and gg.stwr == gs.stwr else 0.0
            scores[i, j] = overlap + form_bonus + stwr_bonus
    row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
    return col_ind, row_ind

def evaluate_group(sys: Group, gold: Group, group_eval: ScoreTracker):
    score_spans(gold.msg, sys.msg, group_eval.tp_parts["quote"], group_eval.fp_parts["quote"], group_eval.fn_parts["quote"])
    if sys.form == gold.form:
        group_eval.tp_types[gold.form].append(group_eval.fid)
    else:
        if sys.form is not None:
            group_eval.fp_types[sys.form].append(group_eval.fid)
        if gold.form is not None:
            group_eval.fn_types[gold.form].append(group_eval.fid)

    if sys.stwr == gold.stwr:
        group_eval.tp_mediums[gold.stwr].append(group_eval.fid)
    else:
        if sys.stwr is not None:
            group_eval.fp_mediums[sys.stwr].append(group_eval.fid)
        if gold.stwr is not None:
            group_eval.fn_mediums[gold.stwr].append(group_eval.fid)
    
    for role in group_eval.parts[1:]:
        score_spans(gold.roles[role], sys.roles.get(role, []), group_eval.tp_parts[role], group_eval.fp_parts[role], group_eval.fn_parts[role])





def get_documents(file_or_folder):
    """Takes a list of files and returns annotations. """

    documents = {}
    if os.path.isdir(file_or_folder):
        for filename in os.listdir(file_or_folder):            
            if filename.endswith(".json"):
                with open(os.path.join(file_or_folder, filename)) as f:
                    js = json.load(f)
                doc = Document(js["documentName"], js["annotations"])
                documents[doc.id] = doc
    else:
        with open(file_or_folder) as f:
            for line in f:
                js = json.loads(line)
                doc = Document(js["documentName"], js["annotations"])
                documents[doc.id] = doc

    return documents


def evaluate(prediction, gold, print_scores=True):
    gold_docs = get_documents(gold) if isinstance(gold, str) else gold
    predict_docs = get_documents(prediction) if isinstance(prediction, str) else prediction
    e = Evaluator(predict_docs, gold_docs)
    if print_scores:
        e.print_report(sys.stdout)
    return e.scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--prediction", type=str, help="")
    parser.add_argument("--gold", type=str, help="")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(evaluate(args.prediction, args.gold))
