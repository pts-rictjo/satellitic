lic_ = """
   Copyright 2025 Richard Tjörnhammar

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
from .global import *

desc_ = """ -----------------------
 Utilities: TLE download / parsing
 -----------------------"""

def fetch_tle_group_celestrak(group: str, timeout: int = 30) -> str:
    url = f"https://celestrak.com/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def parse_tle_text(raw: str) -> List[Tuple[str,str,str]]:
    """
    Parse TLE text (3-line blocks: name, line1, line2).
    Returns list of (name, line1, line2).
    Robust to extra blank lines.
    """
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip() != ""]
    tles = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i].strip()
        l1 = lines[i+1].strip()
        l2 = lines[i+2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            tles.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return tles

def load_local_tles(filepath: str) -> List[Tuple[str,str,str]]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
    return parse_tle_text(data)


def download_tle_data (
    out_dir: str = "tle_downloads",
    groups: List[str] = ALL_CELESTRAK_GROUPS,
):
    os.makedirs(out_dir, exist_ok=True)
    # gather TLEs from CelesTrak primary
    tles  = []
    names = []

    for g in groups:
        try:
            print(f"Fetching TLEs for group '{g}' from CelesTrak...")
            raw = fetch_tle_group_celestrak(g)
            tles_group = parse_tle_text(raw)
            names.append( f"{out_dir+'/'}{g}TLE.txt" )
            print(f"  parsed {len(tles_group)} TLEs from {g}")
            fo = open(f"{out_dir+'/'}{g}TLE.txt","w")
            print ( raw , file=fo )
            fo.close()
            tles.extend(tles_group)
        except Exception as e:
            print(f"  failed to fetch {g} from CelesTrak: {e}; continuing")

def gather_tle_data (
    out_dir: str = "tle_downloads",
    groups: List[str] = ALL_CELESTRAK_GROUPS,
    local_tle_file: str = "tle_local.txt",
):
    os.makedirs(out_dir, exist_ok=True)
    # gather TLEs from CelesTrak primary
    tles  = []
    names = []

    for g in groups:
        try:
            print(f"Fetching TLEs for group '{g}' from CelesTrak...")
            raw = fetch_tle_group_celestrak(g)
            tles_group = parse_tle_text(raw)
            names.append( f"{out_dir+'/'}{g}TLE.txt" )
            print(f"  parsed {len(tles_group)} TLEs from {g}")
            fo = open(f"{out_dir+'/'}{g}TLE.txt","w")
            print ( raw , file=fo )
            fo.close()
            tles.extend(tles_group)
        except Exception as e:
            print(f"  failed to fetch {g} from CelesTrak: {e}; continuing")
    fo = open(local_tle_file,"w")
    fo .close()
    fo = open(local_tle_file,"a")
    for name in names :
        print ( name , ':', os.path.getsize(name) )
        with open(name,"r") as input :
            try:
                for line in input :
                    if len(line.replace(" ","").replace("\n","")) > 0 :
                        print ( line.replace( "\n" , "" ) , file=fo )
            except Exception as err:
                continue
    fo.close()

def collate_tle_data(
    out_dir: str        = "tle_downloads",
    groups: List[str]   = ALL_CELESTRAK_GROUPS,
    local_tle_file: str = "tle_local.txt",
):
    os.makedirs(out_dir, exist_ok=True)
    # gather TLEs from CelesTrak primary
    tles  = []
    names = []
    fo = open(local_tle_file,"w")
    fo .close()
    fo = open(local_tle_file,"a")
    for g in groups :
        name = f"{out_dir+'/'}{g}TLE.txt"
        print ( name , ':', os.path.getsize(name) )
        with open(name,"r") as input :
            try:
                for line in input :
                    if len(line.replace(" ","").replace("\n","")) > 0 :
                        print ( line.replace( "\n" , "" ) , file=fo )
            except Exception as err:
                continue
    fo.close()

if __name__ == "__main__":
    if True :
        download_tle_data()
        collate_tle_data()
