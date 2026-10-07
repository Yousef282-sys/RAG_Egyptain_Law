from src.ingestion.parser import parse_pages_to_articles

# All text below is small SYNTHETIC test data written for this test suite,
# structured like the real PDF: article heading on its own line, body text
# on following line(s). NOT real Egyptian Civil Code content.


def test_parses_single_simple_article():
    pages = [(1, "المادة 1\nنص تجريبي بسيط لهذه المادة.")]
    articles = parse_pages_to_articles(pages)
    assert len(articles) == 1
    a = articles[0]
    assert a.article_number == 1
    assert a.source_page == 1
    assert a.is_repealed is False
    assert "تجريبي" in a.text_ar


def test_parses_multiple_articles_across_lines():
    page_text = (
        "المادة 1\n"
        "نص تجريبي للمادة الاولى يمتد على عدة كلمات.\n"
        "تكملة نص المادة الاولى في سطر ثاني.\n"
        "المادة 2\n"
        "نص تجريبي للمادة الثانية.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert [a.article_number for a in articles] == [1, 2]
    assert "تكملة" in articles[0].text_ar


def test_hierarchy_tracking_book_chapter_section():
    page_text = (
        "الكتاب الأول تجريبي\n"
        "الباب الأول تجريبي\n"
        "الفصل الأول تجريبي\n"
        "المادة 5\n"
        "نص تجريبي.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    a = articles[0]
    assert a.book is not None and "الكتاب" in a.book
    assert a.chapter is not None and "الباب" in a.chapter
    assert a.section is not None and "الفصل" in a.section


def test_new_book_resets_chapter_and_section():
    page_text = (
        "الباب الأول تجريبي\n"
        "الفصل الأول تجريبي\n"
        "المادة 1\n"
        "نص تجريبي.\n"
        "الكتاب الثاني تجريبي\n"
        "المادة 2\n"
        "نص تجريبي.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert articles[0].chapter is not None
    assert articles[0].section is not None
    # after a new "book" header, chapter/section context resets
    assert articles[1].chapter is None
    assert articles[1].section is None


def test_repealed_article_is_flagged():
    page_text = "المادة 3\nهذه المادة ملغاة بموجب قانون تجريبي.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    assert articles[0].is_repealed is True


def test_articles_span_multiple_pages():
    pages = [
        (1, "المادة 1\nنص تجريبي.\n"),
        (2, "المادة 2\nنص تجريبي على الصفحة الثانية.\n"),
    ]
    articles = parse_pages_to_articles(pages)
    assert articles[0].source_page == 1
    assert articles[1].source_page == 2


def test_bilingual_line_split_into_text_en():
    page_text = (
        "المادة 1\n"
        "نص تجريبي بالعربية.\n"
        "This is synthetic English test text for article one.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert articles[0].text_en is not None
    assert "synthetic" in articles[0].text_en


def test_citation_includes_hierarchy_and_number():
    page_text = "الباب الأول تجريبي\nالمادة 7\nنص تجريبي.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    assert "7" in articles[0].citation
    assert "الباب" in articles[0].citation


def test_real_pdf_style_heading_with_english_duplicate():
    # mirrors the real PDF's bilingual heading line, e.g. "Article 244 مادة 244"
    page_text = "Article 5 مادة 5\nنص تجريبي لجسم المادة.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 5


def test_midsentence_citation_is_not_a_new_article():
    # regression test: a cross-reference to another article inside a long
    # sentence must NOT be treated as the start of a new article (mirrors
    # the real PDF's "...in accordance with the provisions of Article 1119..."
    # citation inside Article 247's body).
    page_text = (
        "المادة 1\n"
        "نص تجريبي طويل يحتوي على اشارة الي المادة 99 ضمن الجملة نفسها "
        "ولا يجب ان يعتبر هذا بداية مادة جديدة في هذا الاختبار التجريبي.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 1


def test_wrapped_short_citation_line_is_not_a_new_article():
    # regression test: a short, wrapped citation line with no English
    # "Article N" duplicate (mirrors real PDF lines like "طبقا للمادة 440"
    # or ".عليه فى المادة 914", which are line-wrap remainders of a longer
    # sentence, not headings) must not start a new article.
    page_text = (
        "المادة 1\n"
        "نص تجريبي يشير الى حكم اخر بموجبه\n"
        "طبقا للمادة 99\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 1


def test_long_sentence_mentioning_chapter_word_is_not_a_hierarchy_reset():
    # regression test: mirrors real PDF sentences like "قلم الكتاب" (court
    # clerk's office) and "في هذا الباب" (in this Chapter, a cross-
    # reference) -- mentioning the word in a long sentence must not reset
    # hierarchy state or get treated as a heading.
    page_text = (
        "BOOK I الكتاب الأول\n"
        "المادة 1\n"
        "نص تجريبي طويل يذكر عبارة قلم الكتاب ضمن جملة كاملة وطويلة جدا "
        "ولا يجب ان يفسر هذا كعنوان كتاب جديد فى هذا الاختبار التجريبي.\n"
        "المادة 2\n"
        "نص تجريبي ثاني.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert articles[0].book is not None and "الكتاب" in articles[0].book
    assert articles[1].book == articles[0].book
    # regression test: real PDF hierarchy headers are bilingual on one
    # line, e.g. "BOOK II الكتاب الثاني" -- must still update hierarchy
    # state and must NOT be absorbed into the previous article's body.
    page_text = (
        "المادة 1\n"
        "نص تجريبي.\n"
        "BOOK II الكتاب الثاني\n"
        "المادة 2\n"
        "نص تجريبي ثاني.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 2
    assert "الكتاب" not in articles[0].text_ar
    assert articles[1].book is not None and "الكتاب" in articles[1].book


def test_range_repeal_notice_is_skipped_not_absorbed():
    # regression test: mirrors the real PDF's "Articles 389-417 repealed
    # المواد من ٣٨٩ إلى ٤١٧ ملغاة" notice, which must not pollute the
    # currently-open article's text or wrongly flip its is_repealed flag.
    page_text = (
        "المادة 1\n"
        "نص تجريبي عادي غير ملغي.\n"
        "Articles 2-5 repealed المواد من 2 إلى 5 ملغاة\n"
        "المادة 6\n"
        "نص تجريبي سادس.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert [a.article_number for a in articles] == [1, 6]
    assert articles[0].is_repealed is False
    assert "المواد" not in articles[0].text_ar


def test_bilingual_heading_with_mismatched_number_rejected():
    # if the English and Arabic numbers on a heading-shaped line disagree,
    # it is not trusted as a clean heading (defensive regression case).
    page_text = "المادة 1\nنص تجريبي.\nArticle 7 مادة 8\nنص اخر.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    # the mismatched line 3 is not treated as a new heading; its text is
    # appended to the still-open article 1 instead.
    assert [a.article_number for a in articles] == [1]


def test_bilingual_heading_with_trailing_english_sentence_before_arabic_marker():
    # regression test: mirrors real PDF headings like "Article 277 If the
    # option belongs to the debtor and not مادة ٢٧٧", where a real English
    # sentence fragment sits between the English and Arabic markers on
    # the same line. Must still be recognized as a heading (not rejected
    # as if it were a mid-sentence citation), and the sentence fragment
    # must be preserved as real body content, not discarded.
    page_text = "Article 9 If the option belongs to the debtor مادة 9\nنص تجريبي.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 9
    assert "option belongs to the debtor" in articles[0].text_en


def test_english_heading_missing_leading_letter_is_recognized():
    # regression test: mirrors a real PDF extraction glitch where the
    # leading "A" of "Article" was dropped ("rticle 452 مادة ٤٥٢").
    page_text = "rticle 9 مادة 9\nنص تجريبي.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 9


def test_english_only_heading_with_no_arabic_marker_at_all():
    # regression test: mirrors the real PDF's "Article1022" heading,
    # which has NO Arabic marker pairing on the same line at all.
    page_text = "Article1022\nIn the absence of an agreement to the contrary.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 1022


def test_wrapped_arabic_only_citation_with_trailing_punctuation_rejected():
    # regression test: the exact residual bug from the previous spot
    # check -- a short, English-free wrapped citation line with ONLY a
    # trailing period as leftover (".المادة 1148") must still be rejected
    # as a heading (it is not the *entire* line once even a bare period
    # is counted).
    page_text = (
        "المادة 1\n"
        "نص تجريبي يشير الى حكم مطول فى نهايته\n"
        ".المادة 1\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert len(articles) == 1
    assert articles[0].article_number == 1


def test_heading_marker_text_not_included_in_body():
    # regression test: the heading itself ("Article 9 مادة 9") must not
    # leak into text_ar/text_en; article_number already captures it.
    page_text = "Article 9 مادة 9\nنص تجريبي بسيط.\n"
    articles = parse_pages_to_articles([(1, page_text)])
    a = articles[0]
    assert "مادة" not in a.text_ar
    assert a.text_en is None or "Article" not in a.text_en
    assert "تجريبي" in a.text_ar


def test_midsentence_bilingual_cross_reference_is_not_a_new_article():
    # regression test: mirrors the real PDF's "with Article 298 claim
    # from the co-debtor who has been .للمادة 298" -- a cross-reference
    # citing the SAME article number in both English and Arabic on one
    # line, but with real sentence text BEFORE the English marker. Must
    # not be treated as article 298's heading (that heading is the
    # separate, later, clean "Article 298 مادة 298" line).
    page_text = (
        "المادة 1\n"
        "نص تجريبي يشير الى حكم اخر\n"
        "with Article 9 claim from the co-debtor who has been .للمادة 9\n"
        "المادة 9\n"
        "نص تجريبي حقيقي للمادة التاسعة.\n"
    )
    articles = parse_pages_to_articles([(1, page_text)])
    assert [a.article_number for a in articles] == [1, 9]
    assert "التاسعة" in articles[1].text_ar
