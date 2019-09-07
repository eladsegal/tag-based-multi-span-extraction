
class CleaningObejective(object):

    whitelist = []

    def is_fitting_objective(self, passage, question, answer):
        raise NotImplementedError

    # expected output is a dictionary with the changed elements. If the answer only was changed then {'answer': new_answer_value}
    def clean(self, passage, question, answer, passage_tagging, question_tagging):
        raise NotImplementedError

