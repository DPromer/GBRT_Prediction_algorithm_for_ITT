# -----------------------------------------------------------------------------------
# abaqus_nanoindentation.py
#
# Author: Darren Promer
# Written for fulfilment of ME700: Master's Thesis
# at Western Michigan University
# -----------------------------------------------------------------------------------

"""
This is a python program written to automate the iterative methodology proposed in my Thesis:

Determination of Mechanical Properties through Nanoindentation and Finite Element Simulation

It will house all of the functions which require ABAQUS packages and which require being called from 'abaqus python' at the command line interface

Typically this is be imported as:

import abaqus_nanoindentation as abq_ni

REV 1.0
"""
# -----------------------------------------------------------------------------------
## COMMONPLACE MODULES
# -----------------------------------------------------------------------------------
from odbAccess import openOdb
from abapy.misc import dump
from abapy.postproc import GetHistoryOutputByKey as gho

# -----------------------------------------------------------------------------------
def pcklcreate(workdir, name):   
    # ABAQUS/PYTHON POST PROCESSING SCRIPT
    # Run using abaqus python / abaqus viewer -noGUI / abaqus cae -noGUI
    print("Initiation of pckl creation: "+name+".pckl")
    print
    
    # Opening the Odb File
    odb = openOdb(workdir + '/' + name + '.odb')
    print("odb = openOdb(workdir + '/' + name + '.odb')")

    # Finding back the position of the reference node of the indenter. Its number is stored inside a node set named REF_NODE.

    ref_node_label = odb.rootAssembly.instances['I_INDENTER'].nodeSets['RP_INDENTER'].nodes[0].label
    print("ref_node_label = odb.rootAssembly.instances['I_INDENTER'].nodeSets['RP_INDENTER'].nodes[0].label")

    # Getting back the reaction forces along Y (RF2) and displacements along Y (U2) where they are recorded.
    RF2 = gho(odb, 'RF2')
    U2  = gho(odb, 'U2')
    print("RF2 = gho(odb, 'RF2')")
    print("U2  = gho(odb, 'U2')")

    # Packing data
    data = {'ref_node_label': ref_node_label, 'RF2':RF2, 'U2':U2}
    print("data = {'ref_node_label': ref_node_label, 'RF2':RF2, 'U2':U2}")

    # Dumping data
    dump(data, workdir + '/' + name + '.pckl')
    print("dump(data, workdir + '/' + name + '.pckl')")

    # Closing Odb
    odb.close()
    print("odb.close()")

    print
    print("ERROR REPORT:")