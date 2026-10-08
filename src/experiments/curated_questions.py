"""Curated evaluation questions for the Egyptian Civil Code corpus.

Each entry: (article_number, language, question). The question was written
by reading that article's own text in data/processed/civil_code.json, and
asks only about something the article itself states; the article is the
single expected/relevant source. NO legal content from outside the corpus
is used. The `reference` answer for each question is NOT written here -- it
is extracted programmatically from the corpus (see
src/experiments/eval_set.py: english_extract), so ground truth can never
drift from, or be invented beyond, the corpus text.

Only articles whose article number is unique in the corpus (articles 1 and
2 occur twice: Promulgation Law + Code) and whose extracted English reads
as coherent text were used; the corpus' two-column PDF layout scrambles
longer multi-clause articles, so those were deliberately avoided.

Coverage: general provisions, persons, contracts and defects of consent,
obligations and their effects, assignment/extinction, sale, gift,
partnership, compromise, lease, contract for works, services, deposit,
annuity, insurance, suretyship, ownership, co-ownership, possession,
servitudes and pledge.
"""
from __future__ import annotations

from typing import List, Tuple

CURATED_QUESTIONS: List[Tuple[int, str, str]] = [
    # ---- English -------------------------------------------------------
    (3, "en", "How are periods of limitation calculated, unless a law expressly provides otherwise?"),
    (38, "en", "What names must every person have, and to whom is the family name passed?"),
    (50, "en", "What can a person demand when rights inherent in his personality have been unlawfully infringed?"),
    (85, "en", "What are fungible things under the Civil Code?"),
    (96, "en", "What is the legal effect of an acceptance that goes beyond the offer or is accompanied by a restriction or modification?"),
    (105, "en", "In whose favor are the rights and obligations when a representative concludes a contract within the limits of his authority in the name of his principal?"),
    (120, "en", "In what circumstances may a party to a contract demand its avoidance for an essential mistake?"),
    (136, "en", "When is a contract void because of the consideration for an obligation?"),
    (143, "en", "What happens when only part of a contract is void or voidable?"),
    (159, "en", "What happens to the correlative obligations when an obligation under a bilateral contract is extinguished by impossibility of performance?"),
    (168, "en", "What is a person responsible for who causes injury to another in order to avoid a greater injury threatening himself or a third party?"),
    (182, "en", "When may a payment that was not due be recovered?"),
    (201, "en", "Can a debtor claim restitution of what he voluntarily gave to discharge a natural obligation?"),
    (216, "en", "May the judge reduce damages if the creditor, by his own fault, contributed to the loss?"),
    (279, "en", "Is solidarity between creditors or between debtors presumed, and how is it created?"),
    (341, "en", "Can a creditor be forced to accept something other than the very thing due, even if its value is equal or greater?"),
    (371, "en", "How are obligations extinguished by a voluntary release of the debtor, and when is the release completed?"),
    (431, "en", "In what state is the vendor bound to deliver the thing sold to the purchaser?"),
    (437, "en", "What happens if the thing sold perishes before delivery because of a cause beyond the vendor's control?"),
    (482, "en", "How does the Civil Code define exchange?"),
    (549, "en", "What is a compromise under the Civil Code?"),
    (585, "en", "What must a lessee do about matters that require the lessor's intervention, such as urgent repairs or defects?"),
    (595, "en", "Who remains guarantor for the performance of the assignee's obligations when a lease is assigned?"),
    (664, "en", "What happens to a contract for works if performance of the work becomes impossible?"),
    (681, "en", "When is the performance of services presumed to be made for remuneration?"),
    (773, "en", "How can suretyship be established, even if the principal obligation can be proved by oral evidence?"),
    (870, "en", "Who acquires ownership of a movable that has no owner?"),
    (964, "en", "What is presumed about the possessor of a right?"),
    # ---- Arabic --------------------------------------------------------
    (15, "ar", "ما القانون الذي يحكم الالتزام بالنفقة بين الأقارب؟"),
    (32, "ar", "ما الأحكام التي تسري على المفقود والغائب؟"),
    (113, "ar", "من يملك توقيع الحجر على المجنون والسفيه ورفعه عنهما؟"),
    (190, "ar", "متى تسري قواعد الوكالة على الفضالة؟"),
    (206, "ar", "ماذا يتضمن الالتزام بنقل حق عيني؟"),
    (225, "ar", "هل يجوز للدائن المطالبة بتعويض يزيد على المبلغ المحدد في العقد عندما تتجاوز الخسارة هذا المبلغ؟"),
    (262, "ar", "متى تنتهي حالة الإعسار من تلقاء نفسها؟"),
    (307, "ar", "ما الذي تشمله حوالة الحق؟"),
    (315, "ar", "كيف تتم حوالة الدين؟"),
    (454, "ar", "هل يوجد ضمان للعيوب في البيع القضائي أو البيع الإداري بالمزاد؟"),
    (473, "ar", "ما مدى ضمان من يبيع تركة دون بيان تفاصيل محتوياتها؟"),
    (491, "ar", "ما الأحكام التي تسري على هبة شيء معين بالذات لا يملكه الواهب؟"),
    (498, "ar", "ما حدود التزام الموهوب له بتنفيذ العوض إذا تبين أن قيمة الهبة أقل من قيمة العوض المشروط؟"),
    (533, "ar", "ماذا يحدث لسلطات المديرين وللشخصية الاعتبارية للشركة عند حلها؟"),
    (578, "ar", "ما أثر الاتفاق على استبعاد ضمان التعرض أو العيوب أو الحد منه إذا أخفى المؤجر سببه غشا؟"),
    (621, "ar", "ما مدة المزارعة إذا لم تحدد لها مدة؟"),
    (626, "ar", "هل تنتهي المزارعة بوفاة المؤجر أو بوفاة المستأجر؟"),
    (721, "ar", "هل يجوز للمودع عنده أن يعين شخصا آخر لحفظ الشيء المودع دون إذن المودع؟"),
    (726, "ar", "متى يعتبر عقد الوديعة قرضا؟"),
    (744, "ar", "متى يصح اشتراط عدم قابلية الإيراد المرتب مدى الحياة للحجز؟"),
    (751, "ar", "إلى أي مدى يلتزم المؤمن بتعويض المؤمن له؟"),
    (767, "ar", "هل يضمن المؤمن الأضرار الناشئة عن الحريق إذا نشب بسبب عيب في الشيء المؤمن عليه؟"),
    (783, "ar", "ما أثر قبول الدائن شيئا آخر وفاء للدين على ذمة الكفيل؟"),
    (813, "ar", "ما حق المالك في إلزام جيرانه بوضع علامات الحدود، وكيف تقسم نفقات ذلك؟"),
    (830, "ar", "هل يجوز لأحد الشركاء في الشيوع اتخاذ إجراءات حفظ المال الشائع دون موافقة الباقين؟"),
    (929, "ar", "لمن تكون ملكية المنشآت الخفيفة التي تقام على أرض الغير دون قصد بقائها بصفة دائمة؟"),
    (950, "ar", "كيف يكتسب عديم التمييز الحيازة؟"),
    (1016, "ar", "كيف يكتسب حق الارتفاق؟"),
    (1097, "ar", "ما الأموال التي يجوز أن تكون محلا للرهن الحيازي؟"),
]
