import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def text_to_tags(txt):
    tags = []
    words = []
    label2tag = {'LABEL_0': 'other', 'LABEL_1': 'mountain'}
    res = classifier(txt)
    word = ''
    for elem in res:
        if elem['word'][0] != '#':
            tags.append(label2tag[elem['entity']])
            if word != '':
                words.append(word)
                word = ''
            word += elem['word']
        else:
            word += elem['word'][2:]
    words.append(word)
    return {word: tag for word, tag in zip(words, tags)}


def main():
    parser = argparse.ArgumentParser(description='Perform Mountain Name Recognition on input text.')
    parser.add_argument('--text', type=str, required=True, help='Input text for NER.')

    args = parser.parse_args()

    text = args.text

    result = text_to_tags(text)
    print(result)


if __name__ == "__main__":
    large_model = AutoModelForTokenClassification.from_pretrained("dieumerci/mountain-recognition-ner")
    tokenizer = AutoTokenizer.from_pretrained("dieumerci/mountain-recognition-ner")
    classifier = pipeline("ner", model=large_model, tokenizer=tokenizer)

    main()
