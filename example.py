# import dotenv
#
# dotenv.load_dotenv()
import dspy
import json

from dspy import Retry
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from ouroboros import Ouroboros

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=400)
dspy.settings.configure(lm=turbo)


def format_checker(choice_string):
    try:
        json.loads(choice_string)
        return True
    except:
        return False


def is_correct_answer_included(answer, choice_string):
    return answer in choice_string


def is_plausibility_yes(assessment_answer):
    return "yes" in assessment_answer.lower()


class QuizChoiceGenerationWithAssertions(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(
            "question, correct_answer, number_of_choices -> answer_choices")
        # has specified instruction to guide inputs -> outputs

    def forward(self, question, answer):
        choice_string = self.generate_choices(question=question, correct_answer=answer,
                                              number_of_choices="4").answer_choices
        dspy.Suggest(format_checker(choice_string),
                     "The format of the answer choices should be in JSON format. Please revise accordingly.")
        dspy.Suggest(is_correct_answer_included(answer, choice_string),
                     "The answer choices do not include the correct answer to the question. Please revise accordingly.")
        plausibility_question = ('Are the distractors in the answer choices plausible and not easily identifiable as '
                                 'incorrect? Reply with "Yes" or "No"')
        plausibility_assessment = dspy.Predict("question, answer_choices, assessment_question -> assessment_answer"
                                               )(question=question, answer_choices=choice_string,
                                                 assessment_question=plausibility_question)
        dspy.Suggest(is_plausibility_yes(plausibility_assessment.assessment_answer), "The answer choices are not "
                                                                                     "plausible distractors or are "
                                                                                     "too easily identifiable as "
                                                                                     "incorrect. Please revise to "
                                                                                     "provide more challenging and "
                                                                                     "plausible distractors.")
        return dspy.Prediction(choices=choice_string)


quiz_choice_with_assertion = assert_transform_module(QuizChoiceGenerationWithAssertions().map_named_predictors(Retry),
                                                     backtrack_handler)

# print(quiz_choice_with_assertion(
#     question="How long does a FAA first-class medical certificate last for a 41 years old?",
#     answer="6 months"))

effect_our = Ouroboros()


class QuizChoiceGenerationWithEffect(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(
            "question, correct_answer, number_of_choices -> answer_choices")
        self.retries = 0
        self.feedbacks = []
        self.old_output = None
        # has specified instruction to guide inputs -> outputs

    def handle_feedback(self, message):
        self.feedbacks.append(message)
        effect_our.resume()

    def handle_plausible(self, question, choice_string):
        plausibility_question = ('Are the distractors in the answer choices plausible and not easily identifiable as '
                                 'incorrect? Reply with "Yes" or "No"')
        plausibility_assessment = dspy.Predict("question, answer_choices, assessment_question -> assessment_answer"
                                               )(question=question, answer_choices=choice_string,
                                                 assessment_question=plausibility_question)

        if "yes" in plausibility_assessment.assessment_answer:
            self.feedbacks.append("The answer choices are not plausible distractors or are "
                                  "too easily identifiable as incorrect. Please revise to "
                                  "provide more challenging and plausible distractors.")
            effect_our.resume()

    def handle_possible_retry(self, previous_answer):
        if len(self.feedbacks) == 0:
            effect_our.resume()
        self.retries += 1
        if self.retries < 2:
            # TODO Add a way to reset the configs that might have been modified
            dspy.settings.configure(lm=dspy.OpenAI(model='gpt-4o', max_tokens=400))
            self.generate_choices = dspy.ChainOfThought("question, correct_answer, number_of_choices,"
                                                        " old_output, feedback_on_old_output"
                                                        " -> answer_choices")
            effect_our.restart()

    @effect_our.handle(handlers=[("feedback", handle_feedback),
                                 ("plausibility", handle_plausible),
                                 ("possible_retry", handle_possible_retry)])
    def forward(self, question, answer):

        choice_string = self.generate_choices(question=question, correct_answer=answer,
                                              number_of_choices="4", old_output=self.old_output,
                                              feedback_on_old_output=";".join(self.feedbacks)).answer_choices

        if not format_checker(choice_string):
            effect_our.raise_effect("feedback", self, "The format of the answer choices should be in JSON format. "
                                                "Please revise accordingly.")

        effect_our.raise_effect("plausibility", self, question, choice_string)

        effect_our.raise_effect("possible_retry", self,choice_string)

        return dspy.Prediction(choices=choice_string)


print(QuizChoiceGenerationWithEffect()(
    question="How long does a FAA first-class medical certificate last for a 41 years old?",
    answer="6 months"))


# todo converts into a jupiter
# todo integration with DSPy affects usabiliyty
# todo how could things hook up to it generically
    # it seems in dspy this is done internally, but maybe the API could be made to "query" past effects