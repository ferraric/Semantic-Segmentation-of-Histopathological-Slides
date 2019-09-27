import xml.etree.ElementTree as ET
from xml.dom import minidom


"""TODOs
* Adapt some of the path parameters
* Determine scaling parameter (?)

* Check width, height, scale, transform etc.

* To get specific box in image with annotations:
  - scale annotation mask on the svg level
  - convert to png
  - blend with box
"""


class Annotation:
    def __init__(self, mrxs_fname, xml_fname):
        self.box, self.zoom_level = get_source_roi_as_box(xml_fname)


def get_source_roi_as_box(xml_fname):
    """Extracts measures of source region of interest."""
    tree = ET.parse(xml_fname)
    root = tree.getroot()

    source = root[0]
    source_roi = source[0]

    zoom_level = int(source.attrib["zoom_level"])
    offset_left = int(source_roi.attrib["offset_left"])
    offset_top = int(source_roi.attrib["offset_top"])
    width = int(source_roi.attrib["width"])
    height = int(source_roi.attrib["height"])

    box = (offset_left, offset_top, offset_left + width, offset_top + height)

    return box, zoom_level


def xml2svg(xml_fname, svg_template_fname="template.svg",
            svg_output_fname="./output.svg",
            scaling=0.05):
    """Converts annotations from xml as output by 3DHistTech to svg"""

    ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
    ET.register_namespace("svg", "http://www.w3.org/2000/svg")
    ET.register_namespace("cc", "http://creativecommons.org/ns#")
    ET.register_namespace("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    ET.register_namespace("svg", "http://www.w3.org/2000/svg")
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")

    tree = ET.parse(xml_fname)
    root = tree.getroot()

    # Make sure the xml file looks as expected
    format_error_string = "XML file does not have the right format"
    assert root.tag == "annotation_meta_data", format_error_string
    assert root[0].tag == "source", format_error_string
    assert root[1].tag == "destination", format_error_string

    source = root[0]
    source_roi = source[0]  # Check if this really is the roi
    dest = root[1]
    # dest_roi = dest[0]  # Check if this really is the roi

    # Collect annotations
    annotations = []
    for node in dest.iter("annotation"):
        annotations += [node]

    # Parse svg template
    svg_tree = ET.parse(svg_template_fname)
    svg_root = svg_tree.getroot()

    # Set some svg parameters
    svg_root.attrib["width"] = str(int(source_roi.attrib["width"]) * scaling)
    svg_root.attrib["height"] = str(int(source_roi.attrib["height"]) * scaling)
    svg_root.attrib["viewBox"] = "0 0 " + svg_root.attrib["width"] + " " + svg_root.attrib["height"]

    # Convert annotations to svg
    for i, annotation in enumerate(annotations):
        point = annotation[0]
        annotation_str = "m " + str(float(point.attrib["x"]) * scaling) + "," + str(float(point.attrib["y"]) * scaling) + " "
        # annotation_str = "m 0,0 "
        for new_point in annotation[1:]:
            annotation_str += str(float(new_point.attrib["x"]) * scaling - float(point.attrib["x"]) * scaling) + "," + \
                str(float(new_point.attrib["y"]) * scaling - float(point.attrib["y"]) * scaling) + " "
            point = new_point
        annotation_str += " z"

        # new_group = ET.SubElement(svg_root, "g")
        # Append paths to last group
        new_path = ET.SubElement(svg_root[-1], "svg:path")

        # TODO: Check these values / make customizable
        new_path.attrib["style"] = \
            "display:inline;" +\
            "fill:#000000;" +\
            "stroke:#000000;" +\
            "stroke-width:1;" +\
            "stroke-linecap:butt;" +\
            "stroke-linejoin:miter;" +\
            "stroke-miterlimit:4;" +\
            "stroke-dasharray:none;" +\
            "stroke-opacity:1;" +\
            "opacity:1"

        new_path.attrib["d"] = annotation_str
        new_path.attrib["id"] = "path" + str(i).zfill(2)

    svg_tree_str = minidom.parseString(ET.tostring(svg_root, method='xml')).toprettyxml(indent="   ",
                                                                                        newl="")
    with open(svg_output_fname, 'w') as file:
        file.write(svg_tree_str)
