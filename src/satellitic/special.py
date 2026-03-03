def strings_find(a, sub, start=0, end=None):
    """
    En ren Python-version av numpy.strings.find.
    
    a   : en itererbar av strängar
    sub : substring att söka efter
    start, end : valfritt sökområde
    """
    results = []
    for s in a:
        # Python slice hanteras direkt av str.find
        idx = s.find(sub, start, end)
        results.append(idx)

if __name__ == '__main__':
    a = ["hello world", "test string", "abcabc", "no match"]
    print ( strings_find(a, "abc") )
