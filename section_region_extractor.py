import os
import fitz


def match_keyword_related_work(full_text):
    abstract_start_idx = max(full_text.find('\nAbstract\n'), full_text.find('\nABSTRACT\n'))
    intro_start_idx = max(full_text.find('Introduction\n', abstract_start_idx),
                          full_text.find('INTRODUCTION\n', abstract_start_idx))
    if full_text.find('Related Work\n', intro_start_idx) != -1:
        return "Related Work"
    elif full_text.find('Related Works\n', intro_start_idx) != -1:
        return "Related Works"
    elif full_text.find('Related work\n', intro_start_idx) != -1:
        return "Related work"
    elif full_text.find('Related works\n', intro_start_idx) != -1:
        return "Related works"
    elif full_text.find('RELATED WORK\n', intro_start_idx) != -1:
        return "RELATED WORK"
    elif full_text.find('RELATED WORKS\n', intro_start_idx) != -1:
        return "RELATED WORKS"
    elif full_text.find('Background and related work\n', intro_start_idx) != -1:
        return "Background and related work"
    elif full_text.find('Background and related works\n', intro_start_idx) != -1:
        return "Background and related works"
    elif full_text.find('Background\n', intro_start_idx) != -1:
        return "Background"
    elif full_text.find('BACKGROUND\n', intro_start_idx) != -1:
        return "BACKGROUND"
    else:
        return None


def match_keyword_introduction(full_text):
    abstract_start_idx = max(full_text.find('\nAbstract\n'), full_text.find('\nABSTRACT\n'))
    if full_text.find('Introduction\n', abstract_start_idx) != -1:
        return "Introduction"
    elif full_text.find('INTRODUCTION\n', abstract_start_idx) != -1:
        return "INTRODUCTION"
    else:
        return None


def match_keyword_conclusion(full_text):
    abstract_start_idx = max(full_text.find('\nAbstract\n'), full_text.find('\nABSTRACT\n'))
    intro_start_idx = max(full_text.find('Introduction\n', abstract_start_idx),
                          full_text.find('INTRODUCTION\n', abstract_start_idx))
    if full_text.find('Conclusion', intro_start_idx) != -1:
        return "Conclusion"
    elif full_text.find('CONCLUSION', intro_start_idx) != -1:
        return "CONCLUSION"
    else:
        return None


def detect_region_and_annotate(pdf_path, section="related work", output_path=None, save=False):
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return [], "Failed to open PDF file.", "Failed to open PDF file."

    full_text = chr(12).join([page.get_text() for page in pdf_document])
    if section == "related work":
        KEY_WORD = match_keyword_related_work(full_text)
    elif section == "introduction":
        KEY_WORD = match_keyword_introduction(full_text)
    else:
        raise Exception("Error")
    if KEY_WORD is None:
        return [], "None", "None"

    detected_regions = []
    region_started = False
    end_detected = False
    section_font_size = 0.0
    section_font_type = ""
    collected_text = ""
    previous_span = ""
    pass_next = False

    try:
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if 'lines' not in block:
                    continue
                for line_index, line in enumerate(block["lines"]):
                    spans = line["spans"]
                    if len(spans) == 0:
                        continue
                    for span in spans:
                        line_text = span["text"].strip()
                        if not region_started and line_text.find(KEY_WORD) != -1:
                            region_started = True
                            section_font_size = float(span["size"])
                            section_font_type = str(span["font"])
                            bbox = fitz.Rect(block["bbox"])
                            # page.draw_rect(bbox, color=(1, 0, 0), width=1.5)
                            detected_regions.append((page_number, bbox))
                        previous_span = span

                    if region_started:
                        for span in spans:
                            # print(span)
                            if span["text"].find(KEY_WORD) != -1:
                                continue
                            collected_text += span["text"]

                        bbox = fitz.Rect(block["bbox"])
                        # page.draw_rect(bbox, color=(1, 0, 0), width=1.5)
                        detected_regions.append((page_number, bbox))

                        if not end_detected:
                            if not pass_next and abs(float(previous_span["size"]) - section_font_size) < 0.1 and \
                                    str(previous_span["font"]) == section_font_type and \
                                    previous_span["text"].find(KEY_WORD) == -1:
                                if len(previous_span["text"].strip()) >= 3 and \
                                        previous_span["text"].strip()[0].isdigit() and \
                                        previous_span["text"].strip()[1] == "." and \
                                        previous_span["text"].strip()[2].isdigit():
                                    pass_next = True
                                    pass
                                else:
                                    end_detected = True
                                    collected_text = collected_text[:-len(span["text"])]
                                    break
                            else:
                                if pass_next and abs(float(previous_span["size"]) - section_font_size) < 0.1 and \
                                        str(previous_span["font"]) == section_font_type and \
                                        previous_span["text"].find(KEY_WORD) == -1:
                                    pass_next = True
                                else:
                                    pass_next = False

                    collected_text += " "

                if end_detected:
                    break

            if end_detected:
                break

        if detected_regions and save:
            pdf_document.save(output_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], "Error", "Error"


    identifier = False
    exclusive_text = ""
    rem_pages = set()
    for target_region in detected_regions:
        rem_pages.add(target_region[0])
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        if page_number not in rem_pages:
            exclusive_text += page.get_text().strip() + " "
            continue
        blocks = page.get_text("blocks")
        for block in blocks:
            bbox = fitz.Rect(x0=block[0], y0=block[1], x1=block[2], y1=block[3])
            is_draw = True
            for target_region in detected_regions:
                if target_region[0] != page_number:
                    continue
                if bbox.intersects(target_region[1]):
                    is_draw = False
                    exclusive_text += "**[{}]**".format(section) if not identifier else ""
                    identifier = True
                    break

            if is_draw:
                exclusive_text += page.get_textbox(bbox).strip() + " "
                # page.draw_rect(bbox, color=(1, 0, 0), width=1.5)

    if section == "introduction":
        identifier_index = exclusive_text.find("**[{}]**".format(section))
        ref_start_idx = max(exclusive_text.find('References', identifier_index),
                            exclusive_text.find('REFERENCES', identifier_index))
        exclusive_text = exclusive_text[identifier_index: ref_start_idx].replace("**[{}]**".format(section), "").strip().replace('\n', ' ').replace('- ', '')
    elif section == "related work":
        abstract_start_idx = max(exclusive_text.find('Abstract'), exclusive_text.find('ABSTRACT'))
        intro_start_idx = max(exclusive_text.find('Introduction', abstract_start_idx),
                              exclusive_text.find('INTRODUCTION', abstract_start_idx), 0)
        ref_start_idx = max(exclusive_text.find('References', intro_start_idx),
                            exclusive_text.find('REFERENCES', intro_start_idx))
        exclusive_text = exclusive_text[intro_start_idx: ref_start_idx].replace("**[{}]**".format(section), "").strip().replace('\n', ' ').replace('- ', '')
    else:
        raise Exception("Error")

    # print(exclusive_text)
    # pdf_document.save(output_path)

    if not pdf_document.is_closed:
        pdf_document.close()

    if len(detected_regions) == 0:
        return [], "No related work section found.", "No related work section found."
    else:
        return detected_regions, collected_text.strip(), exclusive_text.strip()



