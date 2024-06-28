from fuzzysearch import find_near_matches
from unidecode import unidecode
import string
from tqdm import tqdm
from operator import attrgetter

import io
import pydotplus
import elementpath
from lxml import etree
from copy import deepcopy
from graphviz import Source
from IPython.display import display, Image

xslt_dir = '../../edspdf-train/data/xslt/'

def printXML(element):
    # print(isinstance(element, etree._ElementTree))
    t = element if isinstance(element, etree._ElementTree) else etree.ElementTree(element)
    # t=etree.ElementTree(element)
    str=etree.tostring(t, pretty_print=True, encoding='unicode')
    print(str)
    # return str       
def printXPath(xmltree,xp):
    # r=xmltree.xpath(xp)
    try:
        r=elementpath.select(xmltree,xp)
        print(f'Lenght {len(r)}')
        for x in r:
            printXML(x)
    except:
        print(f"error in expression ({xp}) or inputfile")
def displayXPath(xmltree,xp,pngfile="reponse"):
    # r=xmltree.xpath(xp)
    r=elementpath.select(xmltree,xp)
    reponse=etree.Element("reponse")
    for x in r:
        reponse.append(deepcopy(x))
    displayXMLInline(reponse)
def displayXML(t,pngfile="xpath"):
    f_xslt = open(f"{xslt_dir}xml2dot.xslt","r")
    xslt=etree.parse(f_xslt)
    transform = etree.XSLT(xslt)
    s_dot=transform(t)
    try:
        s=Source(str(s_dot),filename="gendata/"+pngfile+".dot", format="png")
    except:
        print("install graphviz")
        print(etree.tostring(t, pretty_print=True, encoding='unicode'))
    s.view()
def displayXMLInline(t):
    f_xslt = open(f"{xslt_dir}xml2dot.xslt","r")
    xslt=etree.parse(f_xslt)
    transform = etree.XSLT(xslt)
    s_dot=str(transform(t))
    # print(s_dot)
    # with io.StringIO(s_dot) as f:
    #    a=nx.nx_pydot.read_dot(f)
    try:
        dg = pydotplus.graph_from_dot_data(s_dot)
        png = dg.create_png()
        display(Image(png))
    except:
        print("install graphviz")
        print(etree.tostring(t, pretty_print=True, encoding='unicode'))
    # print(s_dot)
    # return s_dot

def get_front_nodes(tree):
    qfront='//front//title-group/* | \
        //front//permissions//p | \
        //front//corresp | \
        //front//aff | \
        //front//contrib-group | \
        //front//abstract/sec | \
        //front//abstract[count(./sec)<1]'
    r = elementpath.select(tree, qfront)

    for i,node in enumerate(r):
        if node.tag == 'contrib-group':
            contrib_group = etree.Element('contrib-groups')
            
            for icontrib,contrib in enumerate(node):
                for icontrib_child,contrib_child in enumerate(contrib):
                    # print(f'icontrib_child {icontrib_child} tag {contrib_child.tag}')
                    if contrib_child.tag != 'name':
                        if contrib_child.tag == 'xref':
                            virgule_av = etree.Element('vigule')
                            virgule_av.text = ','
                            contrib_group.append(virgule_av)
                            
                            contrib_group.append(contrib_child)
                        continue
                    
                    new_name = etree.Element('new-name')
                    surname = None
                    given_names = None
                    for (i2,subname) in enumerate(contrib_child):
                        if subname.tag == 'surname':
                            surname = i2,subname
                        if subname.tag == 'given-names':
                            given_names = i2,subname
                        if surname is not None and given_names is not None:
                            break
                    if surname is not None and given_names is not None:
                        if surname[0]< given_names[0]:
                            new_name.append(given_names[1])
                            new_name.append(surname[1])
                        else:
                            new_name.append(surname[1])
                            new_name.append(given_names[1])
                    # printXML(new_name)
                    contrib_group.append(new_name)
            # printXML(contrib_group)
            r[i] = contrib_group
            break # one contrib group

    # aff
    affs = etree.Element('affs')
    affs_ind = []
    for i,node in enumerate(r):
        if node.tag == 'aff':
            affs_ind.append(i)
            # affs.append(deepcopy(node))
            affs.append(node)
    r[affs_ind[0]] = affs
    r = [n for i,n in enumerate(r) if i not in affs_ind[1:]]
    
    return r

def get_body_nodes(tree):
    qbody='//body//title | //body//p'
    r = elementpath.select(tree, qbody)
    r2 = []

    jump = False
    for i,node in enumerate(r):
        if jump:
            jump = False
            continue
        if node.tag == 'title' and r[1+i].tag == 'p':
            sec = etree.Element('sec')
            sec.append(node)
            sec.append(r[1+i])
            r2.append(sec)
            jump = True
        else:
            r2.append(node)

    return r2

def get_tables_nodes(tree):
    qtables='//table-wrap'
    tables_nodes=elementpath.select(tree, qtables)
    qtable='//label | //caption | //tr | //table-wrap-foot'
    xml_table_nodes = [
        [
            node for node in one_tns if "".join(node.itertext()) != ""
        ]
        for one_tns in [elementpath.select(table_tree, qtable) for table_tree in tables_nodes]
    ]
    # Merging label and caption
    for itable in range(len(xml_table_nodes)):
        label = None
        caption = None
        lc = None
        for ind,node in enumerate(xml_table_nodes[itable]):
            if node.tag == 'label':
                label = (ind,node)
            if node.tag == 'caption':
                caption = (ind,node)
            if label is not None and caption is not None:
                lc = etree.Element('label_caption')
                lc.append(deepcopy(label[1]))
                lc.append(deepcopy(caption[1]))
                xml_table_nodes[itable][label[0]] = lc
                xml_table_nodes[itable] = xml_table_nodes[itable][:caption[0]] + xml_table_nodes[itable][1+caption[0]:]
                break
    # delete other label tags
    table_nodes_ = []
    for table in xml_table_nodes:
        tk_nodes = []
        for node in table:
            if node.tag != 'label':
                tk_nodes.append(node)
        table_nodes_.append(tk_nodes)
    return table_nodes_
    

from pathlib import Path

def get_paths(file_dir):

    file_dir = Path(file_dir)
    xml_path = next(file_dir.glob("*.nxml"))
    pdf_path = next(file_dir.glob("*.pdf"))
    
    return pdf_path, xml_path


class Node():
    def __init__(self, xml_node, id=None, color='', bg_color='',r0='', entropie_threshold=0):
        self.id = id
        self.cpt = 0
        self.xml_node = xml_node
        self.pdf_path = None
        self.xml_path = None
        self.pdf_page_num = None
        self.full = False
        self.text = unidecode(" ".join(xml_node.itertext()))
        self.text_del = delete_white_spaces(self.text)
        self.lblocs: list[Bloc] = []
        self.bloc = None
        self.color = color
        self.bg_color = bg_color
        self.r0 = r0
        self.entropie_threshold = entropie_threshold
    
    def reset(self):
        self.full = False
        self.lblocs = []
        self.bloc = None
    
    def print_node(self, esc=''):
        full = 'complete' if self.full else 'not complete'
        print(f'{esc}Nb blocs {len(self.lblocs)}\n{esc}Node ({self.id} {full}) text: `{self.text}`')
        if self.full:
            print(f'{esc}complete bloc')
            self.bloc.print_bloc(esc=f'{esc}  ')
        else:
            for bloc in self.lblocs:
                print(f'{esc}bloc {bloc.id}')
                bloc.print_bloc(esc=f'{esc}  ')
    
    def matches_tb(self, tb, error_mes=''):
        target_text = delete_white_spaces(unidecode(tb.text))
        if not target_text:
            error_mes = 'empty string of tb'
            return False
        ret = tb_match_(target_text, self.text_del)
        error_mes = 'No error'

        return ret
    
    def merge(self):
        itable, inode = 0, 26
        # if self.r0 == f't{itable}-' and self.id==inode:
        #     print('Avant tri')
        #     self.print_node('  ')
        self.sort()
        # if self.r0 == f't{itable}-' and self.id==inode:
        #     print('\nAprès tri')
        #     self.print_node('  ')
        end = None
        while True:
            # if self.r0 == f't{itable}-' and self.id==inode:
            #     print('Flows')
            imerge1,imerge2 = None,None
            ret_merge1,ret_merge2 = None,None
            tb_merge1,tb_merge2 = None,None
            dmin = None
            ok_m = False
            for ibloc1, bloc1 in enumerate(self.lblocs):    # prec bloc
                bloc1_last_tb, bloc1_last_ret = bloc1.get_last() # last line of prec bloc
                # test if node if full, end of node
                if bloc1_last_ret[-1].end >= len(self.text_del) and \
                    len(bloc1.text) >= len(self.text_del):
                    end = bloc1
                    break
                
                for ibloc2 in range(1+ibloc1, len(self.lblocs)):    # next bloc
                    bloc2 = self.lblocs[ibloc2]
                    bloc2_first_tb, bloc2_first_ret = bloc2.get_first() # first lin of next bloc
                    
                    # bloc1 text short and dist with bloc2 > threshold
                    if len(bloc1.ltb_ret)<2 and len(bloc1.get_first()[0].text) < 10 and \
                        abs(bloc1_last_tb.y1 - bloc2_first_tb.y0) > 0.2: continue
                    
                    fl, (ret_prec, ret_next) = is_following(bloc1_last_ret,bloc2_first_ret)
                    if fl:
                        # if self.r0 == f't{itable}-' and self.id==inode and bloc2_first_tb.page_num==2:
                        #     print(f'\n  ------------------\n')
                        #     print(f'  --> folow entre {ibloc1} et {ibloc2} \n    {bloc1_last_ret} {bloc1_last_tb}\n    {bloc2_first_ret} {bloc2_first_tb}')
                        #     self.lblocs[ibloc1].print_bloc('  ')
                        #     self.lblocs[ibloc2].print_bloc('  ')
                        
                        if imerge1 is None:
                            imerge1,imerge2 = ibloc1,ibloc2
                            ret_merge1,ret_merge2 = ret_prec, ret_next
                            tb_merge1,tb_merge2 = bloc1_last_tb, bloc2_first_tb
                        else:
                            # priorité sur len() plus grande avec bloc2_first_tb à la même hauteur que bloc1_last_tb
                            # abs(tb_merge2.y0 - tb_merge1.y0) > 0.01 and \
                            # if ret_merge2[0].start == ret_next[0].start and \
                            if abs(bloc2_first_tb.y0 - bloc1_last_tb.y0) < abs(tb_merge1.y0 - tb_merge2.y0) and \
                                len(tb_merge2.text)+2 < len(bloc2_first_tb.text):
                                # if self.r0 == f't{itable}-' and self.id==inode and bloc2_first_tb.page_num==2:
                                #     print('  prio sur len()')
                                imerge1,imerge2 = ibloc1,ibloc2
                                ret_merge1,ret_merge2 = ret_prec, ret_next
                                tb_merge1,tb_merge2 = bloc1_last_tb, bloc2_first_tb
                            
                            # priorité sur dist plus petite
                            if ret_merge2[0].start == ret_next[0].start and \
                                ret_next[0].dist < ret_merge2[0].dist:
                                # if self.r0 == f't{itable}-' and self.id==inode and bloc2_first_tb.page_num==2:
                                #     print('  prio sur dist')
                                imerge1,imerge2 = ibloc1,ibloc2
                                ret_merge1,ret_merge2 = ret_prec, ret_next
                                tb_merge1,tb_merge2 = bloc1_last_tb, bloc2_first_tb
                            
                            # priorité dans une page
                            if (tb_merge1.page_num != tb_merge2.page_num) and \
                                (bloc1_last_tb.page_num == bloc2_first_tb.page_num):
                                # if self.r0 == f't{itable}-' and self.id==inode and bloc2_first_tb.page_num==2:
                                #     print('  prio sur page')
                                imerge1,imerge2 = ibloc1,ibloc2
                                ret_merge1,ret_merge2 = ret_prec, ret_next
                                tb_merge1,tb_merge2 = bloc1_last_tb, bloc2_first_tb
                        #
                        # if self.r0 == f't{itable}-' and self.id==inode and bloc2_first_tb.page_num==2:
                        #     print(f'\n  ------------------\n')
                        #     print(f'  <-- prio {imerge1} et {imerge2}')
                        #     self.lblocs[imerge1].print_bloc('  ')
                        #     self.lblocs[imerge2].print_bloc('  ')
                        # taille suffisamment grande len()
                        if len(bloc1.get_first()[0].text) > 20 and len(bloc2_first_tb.text) > 20:
                            imerge1,imerge2 = ibloc1,ibloc2
                            ret_merge1,ret_merge2 = ret_prec, ret_next
                            tb_merge1,tb_merge2 = bloc1_last_tb, bloc2_first_tb
                            ok_m = True
                            break
                if ok_m:
                    break
            if imerge1 is None:
                break   # No more merging posssible
            
            # update ret at merging
            # if self.r0 == f't{itable}-' and self.id==inode:
            #     print(f'\n\nMerge de bloc {imerge1} et {imerge2}')
            #     self.lblocs[imerge1].print_bloc('  ')
            #     self.lblocs[imerge2].print_bloc('  ')
            bloc1_copy = self.lblocs[imerge1].copy_bloc()
            tb_1,ret1 = self.lblocs[imerge1].get_last() # last line of prec block
            tb_2,ret2 = self.lblocs[imerge2].get_first()  # first line of next block
            self.lblocs[imerge1].set_last(tb_1, ret_merge1)
            self.lblocs[imerge2].set_first(tb_2, ret_merge2)

            self.lblocs[imerge1].append_bloc(self.lblocs[imerge2])
            # add copy
            # if len(tb_2.text) <= 2 or
            self.lblocs = self.lblocs[:1+imerge1] + [bloc1_copy] + self.lblocs[1+imerge1:] # [avant,imerge1,copy,apres]
            self.lblocs = self.lblocs[:1+imerge2] + self.lblocs[1+1+imerge2:]
            # if self.r0 == f't{itable}-' and self.id==inode:
            #     print(f'\nAprès merge de {imerge1} et {imerge2}')
            #     self.print_node('  ')
        
        # end of the node, node is full
        # if end is not None:
        #     bmax, min_entropie = end, -1
        # else:
        if end is None:
            bmax, min_entropie = None, float('inf')
            for ibloc, bloc in enumerate(self.lblocs):
                entropie = abs(1 - sum([len(delete_white_spaces(unidecode(tb.text))) for tb,_ in bloc.ltb_ret]) / len(self.text_del))
                if entropie < min_entropie:
                    min_entropie = entropie
                    bmax = bloc
                    if min_entropie <= self.entropie_threshold:
                        end = bloc
                        break
        
        if end is not None: # min_entropie < self.entropie_threshold: # == -1: # < 0.1
            bmax = end
            self.full = True
            self.bloc = bmax    # [(tb,ret) for tb,ret in bmax.ltb_ret]
            for tb,_ in bmax.ltb_ret:
                tb.inode = self.id
            # color
            for rank,(tb,ret) in enumerate(bmax.ltb_ret):
                tb.rank = f'{self.r0}{self.id}-{rank}'
                tb.color = self.color
                tb.bg_color = self.bg_color

                    
    
    def filter_blocs(self):
        new_lblocs = []
        for bloc in self.lblocs:
            bloc.filter_tbs()
            if len(bloc.ltb_ret) > 0:
                new_lblocs.append(bloc)
        
        self.lblocs = new_lblocs
    
    def sort(self):
        def sorted_by_first_tb(bloc):
            first_tb,first_ret = bloc.get_first()
            min_start = first_ret[0].start
            
            return first_tb.page_num, min_start, first_tb.y0
        self.lblocs = sorted(self.lblocs, key=lambda bloc: sorted_by_first_tb(bloc))
    
    def add_bloc(self, bloc):
        self.lblocs.append(bloc)
    
    def add_tb(self, tb, ret, mes=''):
        bloc = None
        for b in self.lblocs:
            can, (last_ret_fl, ret_fl) = b.can_append_tb_ret(tb, ret)
            if can:
                bloc = b
                break
        
        if bloc is None:
            mes = f'No follow bloc for tb:' + \
                  f'\n  text: `{unidecode(tb.text)}`' + \
                  f'\n  ret : {ret}'
            new_bloc = Bloc([(tb,ret)], id=self.cpt)
            self.cpt += 1
            self.lblocs.append(new_bloc)
        else:
            last_tb, last_ret = bloc.get_last()
            b.ltb_ret[-1] = (last_tb, last_ret_fl)
            mes = f'Follow bloc' + \
                f'\n  last tb text: `{unidecode(last_tb.text)}`' + \
                f'\n  last ret {last_ret}' + \
                f'\n  tb text: `{unidecode(tb.text)}`' + \
                f'\n  ret {ret}'
            
            bloc.add_tb_ret(tb,ret_fl)
    
    def del_bloc(self, i):
        if (0<=i) and (i<len(self.lblocs)):
            self.lblocs = self.lblocs[0:i] + self.lblocs[1+i:]
        
class Bloc():
    def __init__(self, ltb_ret=[], id=None):
        self.id = id
        self.ltb_ret = [(tb,ret) for tb,ret in ltb_ret]
        self.text = "".join([unidecode(tb.text) for tb,ret in self.ltb_ret])
    
    def filter_bloc(self, page_num):
        self.ltb_ret = list(filter(lambda tb_ret: tb_ret[0].page_num == page_num, self.ltb_ret))
    
    def print_bloc(self, esc=''):
        text = " - ".join([unidecode(tb.text) for tb,ret in self.ltb_ret])
        print(f'{esc}Id {self.id} nb tbs {len(self.ltb_ret)}\n{esc}Text: `{text}`')
        for rank, (tb,ret) in enumerate(self.ltb_ret):
            print(f'{esc}  {(rank):2d} p-{tb.page_num} `{unidecode(tb.text)}` \t\t-> {ret} {tb}')
    
    def get_first(self):
        first = self.ltb_ret[0] if len(self.ltb_ret) > 0 else None
        return first
    def set_first(self, tb, ret):
        self.ltb_ret[0] = (tb, ret)
    
    def get_last(self):
        first = self.ltb_ret[-1] if len(self.ltb_ret) > 0 else None
        return first
    def set_last(self, tb, ret):
        self.ltb_ret[-1] = (tb, ret)
    
    def copy_bloc(self):
        l = [(tb,ret) for tb,ret in self.ltb_ret]
        new_bloc = Bloc(l,id=f'{self.id}*')
        return new_bloc
    
    def add_tb_ret(self, tb, ret):
        self.ltb_ret.append((tb,ret))
        self.text += unidecode(tb.text)
    def append_bloc(self, bloc):
        self.ltb_ret.extend(bloc.ltb_ret)
        self.text += bloc.text
    
    def filter_tbs(self):
        new_bloc = []
        for tb,ret in self.ltb_ret:
            if tb.inode is None:
                new_bloc.append((tb,ret))
        
        self.ltb_ret = new_bloc
    
    def del_tb_ret(self, i):
        if (0<=i) and (i<len(self.ltb_ret)):
            self.ltb_ret = self.ltb_ret[0:i] + self.ltb_ret[1+i:]
    
    def can_append_tb_ret(self, tb, ret, error_mes=''):
        last = self.get_last()
        if last is None: return True
        
        last_tb,last_ret = last
        if len(tb.text) < 10:
            return False, (last_ret, ret)   # creer un new bloc pour une petite ligne
        # same page
        if last_tb.page_num != tb.page_num:
            error_mes = f'page num error:' + \
                        f'\n  last tb page({last_tb.page_num})' + \
                        f'\n  tb page({tb.page_num})'
            return False, (last_ret, ret)
        
        # vertical align
        tb1, tb2 = (last_tb,tb) if (last_tb.x0 <= tb.x0) else (tb,last_tb)
        if (tb2.x0 > tb1.x1):
            first_tb, first_ret = self.get_first()
            tb1_, tb2_ = (first_tb,tb) if (first_tb.x0 <= tb.x0) else (tb,first_tb)
            if (tb2_.x0 > tb1_.x1):
                error_mes = f'vertical align error:' + \
                            f'\n  first tb x0({first_tb.x0}) x1({first_tb.x1})' + \
                            f'\n  last  tb x0({last_tb.x0})  x1({last_tb.x1})' + \
                            f'\n  tb       x0({tb.x0})       x1({tb.x1})'
            return False, (last_ret, ret)
        
        # distance with last tb of the bloc
        tb1, tb2 = (last_tb,tb) if (last_tb.y0 <= tb.y0) else (tb,last_tb)
        if abs(tb2.y0 - tb1.y1) > 0.02:
            error_mes = f'distance with last tb of the bloc error:' + \
                        f'\n  last tb y1({last_tb.y1})' + \
                        f'\n  tb y1({tb.y1})' + \
                        f'\n  dist({abs(tb2.y0 - tb1.y1)})'
            return False, (last_ret, ret)
        
        # is following rets
        fl, (last_ret_fl,ret_fl) = is_following(last_ret, ret)
        if not fl:
            error_mes = f'following rets error:' + \
                        f'  last tb ret({last_ret})' + \
                        f'  ret({ret})'
            return False, (last_ret, ret)
        
        # can append tb with the bloc, append to the last line
        error_mes = 'No error'
        return fl, (last_ret_fl,ret_fl)
            
        
# mapped text boxes
# for i,tb in enumerate(doc.content_boxes):
#     tb.id = i # text box id
# map_tboxes = {tb.id: tb for tb in doc.content_boxes}
# print('Nb text boxes:',len(map_tboxes))

def delete_white_spaces(ch):
    for c in string.whitespace:
        ch = ch.replace(c,"")
    
    return ch

def tb_match_(target_text, source_text, d=0):
    len_target_text = len(target_text)
    # if len_target_text < 4 and len(source_text)>4: # text box that have no text
    #     return False
    
    pdmax = 0.2  if len_target_text<=20 else (\
            0.15 if len_target_text<=40 else \
            0.1)
    dmax = d + int(pdmax * len_target_text)
    
    #match
    ret = find_near_matches(target_text, source_text, max_l_dist=dmax)
    if len(ret)<1: return False

    # if len_target_text <= 4:
    #     end_m = ret[-1].end
    #     if end_m < len(source_text) - 2: return False
    
    # return True if len(ret) > 0 else False
    return ret if len(ret) > 0 else False

whitespaces = '-' + string.whitespace
def is_following(ret_av, ret):
    # if ret[0].start == 0: return False # only one starting tb
    for m_av in ret_av:  # match at the start of the node
        for m in ret:
            if m.start == 0 or m.start == m_av.start: continue # only one starting tb
            # if m.start < m_av.end-m.dist: continue
            if abs(m.start - m_av.end) <= ( max(m_av.dist, m.dist)):
                return True, ([m_av], [m])

    return False, (ret_av, ret)

def get_matches(nodes, dict_page_text_boxes, v=False, ):
    
    for num_page in dict_page_text_boxes:
        page_text_boxes = dict_page_text_boxes[num_page]
        match_page(page_text_boxes, nodes, num_page, v=v)

                
        # page
        
        # check complete nodes
        check_complete_nodes(nodes, num_page, v=v)

        
    # not matched nodes
    if v:
        print(f'\n\nNot matched nodes')
        for inode,node in enumerate(nodes):
            if inode in nodes_ok: continue
            
            print(f'  Node {inode} --(not matched with any block)--\n  Nb blocks {len(blocks)}\n  Text: `{" ".join(node.itertext())}`')

            for iblock,block in enumerate(blocks):
                text_2 = "".join([tb.text for tb,_ in block])
                print(f'  block {iblock}\n    Nb text boxes {len(block)}\n    Text: `{text_2}`')
                for rank2,(tb,_) in enumerate(block):
                    print(f'      `{tb.text}` -> {tb.inode}')
    

def match_page(page_tbs, nodes, num_page, v):
    for tb in page_tbs:
        for node in nodes:
            if node.full:
                continue
            
            # tb matches with node
            ret = node.matches_tb(tb)
            if not ret: continue    # no match
            
            node.add_tb(tb,ret)
            
def match_page_v0(page_text_boxes, nodes, dict_blocs, nodes_ok, num_page, v, r0):
    for tb in page_text_boxes:
        for inode,node in enumerate(nodes):
            if inode in nodes_ok:
                continue
            
            target_text = delete_white_spaces(unidecode(tb.text))
            if not target_text: continue
            source_text = delete_white_spaces(unidecode(" ".join(node.itertext())))
            # print(f'\ntarget `{tb.text}` : `{unidecode(tb.text)}` : `{target_text}`\nnode `{" ".join(node.itertext())}`')
            ret = tb_match_(target_text, source_text)
            if not ret:    # no match
                continue

            val = dict_blocs.get(inode, None)  # (node, blocks)
            if val is None:
                dict_blocs[inode] = (node,[[(tb,ret)]]) # node, list[list[(tb,ret)]]
                continue
            
            blocks = val[1]
            block = None # find_block(tb, blocks)
            for b in blocks:
                tb_av,ret_av = b[-1]
                if inode == -67 and num_page==7:
                    print(f'\nFind bloc `{tb.text}`\n  node text {" ".join(node.itertext())}')
                    print(f'  tb av `{tb_av.text}` {tb_av}')
                    print(f'  tb text `{tb.text}` {tb}')
                # bloc of a single page
                if tb.page_num != tb_av.page_num: continue
                if inode == -67 and num_page==7:
                    print('------page num ok------------')

                tb1, tb2 = (tb_av,tb) if (tb_av.x0 <= tb.x0) else (tb,tb_av)
                # on_line = True if (tb2.x0-0.015<=tb1.x1) else False
                if (tb2.x0 > tb1.x1):
                    tb1_, tb2_ = (b[0][0],tb) if (b[0][0].x0 <= tb.x0) else (tb,b[0][0])
                    if (tb2_.x0 > tb1_.x1): continue
                        # if len(tb.text)>5: continue
                
                if inode == -67 and num_page==7:
                    print('------align ok x0------------')

                tb1, tb2 = (tb_av,tb) if (tb_av.y0 <= tb.y0) else (tb,tb_av)
                if abs(tb2.y0 - tb1.y1) > 0.02: continue
                if inode == -67 and num_page==7:
                    print('------tb2.y0 - tb1.y1 ok------------')

                fl, (ret_av_fl,ret_fl) = is_following(ret_av, ret)
                if fl:
                    block = b
                    b[-1] = (tb_av, ret_av_fl)
                    ret = ret_fl
                    break
            
            if inode == 2:
                print('\n------------------', num_page)
                print(f'\n\nNode ({inode}) text: `{source_text}`\nTb text `{tb.text}`\n  {tb}\nNb blocks {len(blocks)}')
                for iblock,b in enumerate(blocks):
                    text = " | ".join([t.text for t,ret in b])
                    print(f'  block {iblock}\n    Nb text boxes {len(b)}\n    Text: `{text}`')
                    for rank,(t,r) in enumerate(b):
                        print(f'      `{t.text}` --> {t}')#\n      {r}\n      {tb}')
            
            if block is None:
                if inode == 2:
                    print(f'No Follow block\n  `{tb.text}`\n    {ret}')
                new_block = [(tb,ret)]
                blocks.append(new_block)
            else:
                block.append((tb,ret))
                if inode == 2:
                    tb_av,ret_av = block[-2]
                    print(f'Follow bloc ')
                    print(f'  prec tb`{tb_av.text}`\n  {ret_av} - start {ret_av[0].start} - end {ret_av[0].end} - dist {ret_av[0].dist}')
                    print(f'  curr tb`{tb.text}`\n  {ret} - start {ret[0].start} - end {ret[0].end} - dist {ret[0].dist}')
                    # print(f'  m.start - m_av.end = {} <= (1 + m_av.dist + m.dist')
                    for rank,(t,ret) in enumerate(block):
                        print(f'    `{t.text}` --> {t}')#\n      {ret}\n      {tb}')


def check_complete_nodes(nodes, num_page, v):
    for node in nodes:
        if node.full:
            continue
        
        # filter tbs already matched
        node.filter_blocs()
        if len(node.lblocs) < 1:
            continue
        
        node.merge()

def check_complete_nodes_v0(nodes, dict_blocs, nodes_ok, num_page, color, bg_color, v, r0):
    for inode,node in enumerate(nodes):
        if inode in nodes_ok:
            continue
        
        val = dict_blocs.get(inode, None)
        if val is None: continue    # if inode not in dict_blocs
        
        _,blocks = val
        
        # retire les tbs deja matchés avec un noeud
        #  les block dejà matchés
        new_blocks = []
        for block in blocks:
            new_block = []
            for tb,ret in block:
                # tb_inode = getattr(tb, 'inode', None)
                if tb.inode is None:
                    new_block.append((tb,ret))
            if len(new_block) > 0:  # if bloc is not empty
                new_blocks.append(new_block)
        blocks = new_blocks
        
        if len(new_blocks) < 1:
            dict_blocs[inode] = node, []
            continue
            
        source_text = unidecode(" ".join(node.itertext()))
        source_text_del = delete_white_spaces(source_text)

        def sorted_by_first_tb(block):
            first_tb,first_ret = block[0]
            min_start = len(source_text_del)
            
            for m in first_ret:
                if m.start <= min_start:
                    min_start = m.start
            
            return first_tb.page_num, min_start, first_tb.y0
        blocks = sorted(blocks, key=lambda block: sorted_by_first_tb(block))
        
        # before merge blocks
        if inode == 2:
            print(f'\nBefore merge - page {num_page}\n  Node ({inode}) text: `{source_text}`\n  Nb blocks {len(blocks)}')
            for iblock,block in enumerate(blocks):
                text = " | ".join([unidecode(tb.text) for tb,ret in block])
                print(f'  block {iblock}\n  Nb text boxes {len(block)}\n  Text: `{text}`')
                for rank,(tb,ret) in enumerate(block):
                    print(f'    p-{tb.page_num} `{unidecode(tb.text)}` -> {ret} {tb}')#\n      {ret}\n      {tb}')
        
        if inode == 2:
          print('\nMerging')
        end = None
        while True:
            imerge1,imerge2 = None,None
            ret_merge1,ret_merge2 = None,None
            tb_merge1,tb_merge2 = None,None
            dmin = None
            ok_m = False
            for iblock1 in range(len(blocks) ):
                tb_1,ret1 = blocks[iblock1][-1] # last line of prec block
                #end of the node
                if ret1[-1].end >= len(source_text_del): 
                    text = delete_white_spaces("".join([unidecode(tb.text) for tb,ret in blocks[iblock1]]))
                    if len(text) >= len(source_text_del):
                        end = blocks[iblock1]
                        break

                for iblock2 in range(1 + iblock1, len(blocks)):
                    tb_2,ret2 = blocks[iblock2][0]  # first line of next block
                    if len(blocks[iblock1][0][0].text) < 10 and \
                        abs(tb_1.y1 - tb_2.y0) > 0.2: continue
                    
                    fl, (ret1_,ret2_) = is_following(ret1, ret2)
                    
                    if fl:
                        if imerge1 is None:
                            dmin = dist_tb(tb_1, tb_2)
                            imerge1,imerge2 = iblock1,iblock2
                            ret_merge1,ret_merge2 = ret1_, ret2_
                            tb_merge1,tb_merge2 = tb_1, tb_2
                        else:
                            # priorité dans une page
                            if (tb_merge1.page_num != tb_merge2.page_num) and \
                                (tb_1.page_num == tb_2.page_num):
                                imerge1,imerge2 = iblock1,iblock2
                                ret_merge1,ret_merge2 = ret1_, ret2_
                                tb_merge1,tb_merge2 = tb_1, tb_2

                        if len(blocks[iblock1][0][0].text) > 20 and len(tb_2.text) > 20:
                            imerge1,imerge2 = iblock1,iblock2
                            ret_merge1,ret_merge2 = ret1_, ret2_
                            tb_merge1,tb_merge2 = tb_1, tb_2
                            ok_m = True
                            break
                if ok_m: # merge possible between blocks[imerge1]+blocks[imerge2]
                    break
            if imerge1 is None:
                break   # No more merging possible
            
            if inode == 2:
                print(f'\n  Merging block {imerge1} with block {imerge2}')
                # iblock 1
                block = blocks[imerge1]
                text = " | ".join([unidecode(tb.text) for tb,ret in block])
                print(f'    block {imerge1}\n    Nb text boxes {len(block)}\n    Text: `{text}`')
                for rank,(tb,ret) in enumerate(block):
                    print(f'      `{unidecode(tb.text)}` -> {ret} {tb}')#\n      {ret}\n      {tb}')
                # iblock 2
                block = blocks[imerge2]
                text = " | ".join([unidecode(tb.text) for tb,ret in block])
                print(f'    block {imerge2}\n    Nb text boxes {len(block)}\n    Text: `{text}`')
                for rank,(tb,ret) in enumerate(block):
                    print(f'      `{unidecode(tb.text)}` -> {ret} {tb}')#\n      {ret}\n      {tb}')

            # update ret at merging
            tb_1,ret1 = blocks[imerge1][-1] # last line of prec block
            tb_2,ret2 = blocks[imerge2][0]  # first line of next block
            blocks[imerge1][-1] = (tb_1, ret_merge1)
            blocks[imerge2][0] = (tb_2, ret_merge2)

            blocks[imerge1].extend(blocks[imerge2])
            blocks = blocks[:imerge2] + blocks[1+imerge2:]
        
            # after merge
            if inode == 2:
                print(f'\n  After merge block {imerge1} with block {imerge2}\n    Node text: `{source_text}`\n    Nb blocks {len(blocks)}')
                for iblock,block in enumerate(blocks):
                    text = " | ".join([unidecode(tb.text) for tb,ret in block])
                    print(f'    block {iblock}\n      Nb text boxes {len(block)}\n      Text: `{text}`')
                    for rank,(tb,ret) in enumerate(block):
                        print(f'        p-{tb.page_num} `{unidecode(tb.text)}` -> {ret} {tb}')#\n      {ret}\n      {tb}')
        
        # end of the node
        if end is not None:
            bmax, min_entropie = end, -1
            
            # text = "".join([tb.text for tb,_ in bmax])
            # print(f'\nPage {num_page}, Node {inode}  OK END!!')
            # print(f'    Nb text boxes {len(bmax)}')
            # print(f'    Node Text:  `{" ".join(node.itertext())}`')
            # print(f'    Block Text: `{text}`')
            # for rank,(tb,ret) in enumerate(bmax):
            #     print(f'    `{tb.text}` -> {tb.inode} {ret}')
        else:
            bmax, min_entropie = None, float("inf")
            for iblock,block in enumerate(blocks):
                entropie = abs(1 - sum([len(delete_white_spaces(unidecode(tb.text))) for tb,_ in block]) / len(source_text_del))
                if entropie < min_entropie:
                    min_entropie = entropie
                    bmax = block
        
        if min_entropie == -1:  #< 0.1:
            if min_entropie != -1:
                l = len(delete_white_spaces("".join([unidecode(tb.text) for tb,ret in bmax])))
                
                # text = "".join([tb.text for tb,_ in bmax])
                # print(f'\nPage {num_page}, Node {inode}  OK ent!! {l}-{len(source_text_del)}')
                # print(f'    Nb text boxes {len(bmax)}')
                # print(f'    Node Text:  `{" ".join(node.itertext())}`')
                # print(f'    Block Text: `{text}`')
                # for rank,(tb,ret) in enumerate(bmax):
                #     print(f'    `{tb.text}` -> {tb.inode} {ret}')
            
            for tb,_ in bmax:
                tb.inode = inode
            dict_blocs[inode] = node, [tb for tb,_ in bmax]
            nodes_ok[inode] = [tb for tb,_ in bmax]
            
            ####
            if v:
                text = "".join([tb.text for tb,_ in bmax])
                print(f'\nPage {num_page}, Node {inode}  OK !!')
                print(f'    Nb text boxes {len(bmax)}')
                print(f'    Node Text:  `{" ".join(node.itertext())}`')
                print(f'    Block Text: `{text}`')
            for rank,(tb,ret) in enumerate(bmax):
                if v:
                    print(f'    `{tb.text}` -> {tb.inode}')
                # if inode == 12:
                #     print(f'      {ret}')
                tb.rank = f'{r0}{inode}-{rank}'
                tb.color = color
                tb.bg_color = bg_color
            ####
        else:
            dict_blocs[inode] = node,blocks


import numpy as np

def dist_tb(tb_1, tb_2):
    xm1 = (tb_1.x0+tb_1.x1) / 2
    ym1 = (tb_1.y0+tb_1.y1) / 2
    
    xm2 = (tb_2.x0+tb_2.x1) / 2
    ym2 = (tb_2.y0+tb_2.y1) / 2

    de = np.sqrt((xm1-xm2)**2 + (ym1-ym2)**2)
    dy = abs(tb_1.y0 - tb_2.y0)
    return min(de, dy)