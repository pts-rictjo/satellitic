lic_ = """
   Copyright 2026 Richard Tjörnhammar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
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
        idx = str(s).find(sub, start, end)
        results.append(idx)
    return np.array(results)

if __name__ == '__main__':
    a = ["hello world", "test string", "abcabc", "no match"]
    print ( strings_find(a, "abc") )

    a = np.array(["NumPy is a Python library"])
    print( strings_find(a, "Python") )

    print( 'spam, spam, spam'.find('sp', 1, None) )
