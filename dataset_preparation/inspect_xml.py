import glob
import os
import xml.etree.ElementTree as ET

xmls = sorted(glob.glob('/tmp/martian_staging/*.xml'))
print(f"Total XMLs in /tmp/martian_staging: {len(xmls)}")

targets = set()
filters = set()
elevations = []

for x in xmls[:50]:
    try:
        tree = ET.parse(x)
        root = tree.getroot()
        for el in root.iter():
            tag = el.tag.split('}')[-1].lower()
            val = (el.text or '').strip()
            if 'target_name' in tag:
                targets.add(val)
            if 'filter_name' in tag or 'filter_id' in tag:
                filters.add(val)
            if 'mast_elevation' in tag:
                try:
                    elevations.append(float(val))
                except:
                    pass
    except Exception as e:
        pass

print("Unique Targets in sample:", targets)
print("Unique Filters in sample:", filters)
if elevations:
    print(f"Mast Elevation Range in sample: min={min(elevations):.1f} deg, max={max(elevations):.1f} deg")
