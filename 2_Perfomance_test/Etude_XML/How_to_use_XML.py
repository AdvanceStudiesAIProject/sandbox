

#Exemple : https://books.google.fr/books?id=KtcnDwAAQBAJ&pg=PT585&lpg=PT585&dq=reparsed.toprettyxml+doc&source=bl&ots=wzafmQ5Uuj&sig=t2oV_jU56t9yxMegfDunNn5BVoE&hl=pt-PT&sa=X&ved=0ahUKEwjx8vuLjPvZAhXR6qQKHXpMARkQ6AEISzAD#v=onepage&q=reparsed.toprettyxml%20doc&f=false
            #Pretty-Printing XML

import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printedXML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent = " ")

myfile = open("items2.xml", "w")# create a new XML file with the results 
# create the file structure    





frame_number="frame"+str(frame_number)
data = ET.Element("Dataset")  
Frame = ET.SubElement(data, str(frame_number))  
Object = ET.SubElement(Frame, 'Object')
X = ET.SubElement(Object, 'x')
Y = ET.SubElement(Object, 'y') 
H = ET.SubElement(Object, 'h') 
W = ET.SubElement(Object, 'w')    
X.set('Value',str(x))
Y.set('Value',str(y))
H.set('Value',str(h))
W.set('Value',str(w))  
myfile.write(prettify(data)) #write the data on the xml file 

frame_number = frame_number +1



 

 







#FOR PARSING AN XML SITE OF EXEMPLE: https://www.blog.pythonlibrary.org/2010/11/20/python-parsing-xml-with-lxml/
#----------------------------------------------------------------------
# def parseXML(xmlFile):
#     """
#     Parse the xml
#     """
#     f = open(xmlFile)
#     xml = f.read()
#     f.close()
 
#     tree = etree.parse(StringIO(xml))
#     context = etree.iterparse(StringIO(xml))
#     for action, elem in context:
#         if not elem.text:
#             text = "None"
#         else:
#             text = elem.text
#         print(elem.tag + " => " + text)   
 
# if __name__ == "__main__":
#     parseXML("example.xml")