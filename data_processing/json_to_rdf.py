from pathlib import Path
import json
from typing import Any

from py_aas_rdf.models import *

# -- Source: https://colab.research.google.com/drive/14myWROAKKG_0LX_U4stohioPfAIrqGhY?usp=sharing#scrollTo=q_0FWxBcLiO_
# Original Author: Rimaz, the code here is being used!!!

import json
from json.decoder import JSONDecodeError

import random
random.seed(10)

import ipywidgets as widgets
from IPython.display import display

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS

import py_aas_rdf
from py_aas_rdf.models.submodel import Submodel
from py_aas_rdf.models.concept_description import ConceptDescription
from py_aas_rdf.models.asset_administraion_shell import AssetAdministrationShell
import zipfile
import tempfile
from pathlib import Path
import json
from rdflib import Graph

import aas_core3.xmlization as aas_xmlization
import aas_core3.jsonization as aas_jsonization
import os
import io

def bind_namespaces(graph):
    graph.bind("aas-identifiable", Namespace("https://admin-shell.io/aas/3/0/Identifiable/"))
    graph.bind("aas-assetadministrationshell", Namespace("https://admin-shell.io/aas/3/0/AssetAdministrationShell/"))
    graph.bind("aas-assetinformation", Namespace("https://admin-shell.io/aas/3/0/AssetInformation/"))
    graph.bind("aas-assetkind", Namespace("https://admin-shell.io/aas/3/0/AssetKind/"))
    graph.bind("aas-conceptdescription", Namespace("https://admin-shell.io/aas/3/0/ConceptDescription/"))
    graph.bind("aas-dataspecificationiec61360", Namespace("https://admin-shell.io/aas/3/0/DataSpecificationIec61360/"))
    graph.bind("aas-datatypedefxsd", Namespace("https://admin-shell.io/aas/3/0/DataTypeDefXsd/"))
    graph.bind("aas-keytypes", Namespace("https://admin-shell.io/aas/3/0/KeyTypes/"))
    graph.bind("aas-submodel", Namespace("https://admin-shell.io/aas/3/0/Submodel/"))
    graph.bind("aas-specificassetid", Namespace("https://admin-shell.io/aas/3/0/SpecificAssetId/"))
    graph.bind("aas-reference", Namespace("https://admin-shell.io/aas/3/0/Reference/"))
    graph.bind("aas-referencetypes", Namespace("https://admin-shell.io/aas/3/0/ReferenceTypes/"))
    graph.bind("aas-resource", Namespace("https://admin-shell.io/aas/3/0/Resource/"))
    graph.bind("aas-modellingkind", Namespace("https://admin-shell.io/aas/3/0/ModellingKind/"))
    graph.bind("aas-haskind", Namespace("https://admin-shell.io/aas/3/0/HasKind/"))
    graph.bind("aas-hassemantics", Namespace("https://admin-shell.io/aas/3/0/HasSemantics/"))
    graph.bind("aas-referable", Namespace("https://admin-shell.io/aas/3/0/Referable/"))
    graph.bind("aas-property", Namespace("https://admin-shell.io/aas/3/0/Property/"))
    graph.bind("aas-key", Namespace("https://admin-shell.io/aas/3/0/Key/"))
    graph.bind("aas-abstractlangstring", Namespace("https://admin-shell.io/aas/3/0/AbstractLangString/"))
    graph.bind("aas-qualifier", Namespace("https://admin-shell.io/aas/3/0/Qualifier/"))
    graph.bind("aas-administrativeinformation", Namespace("https://admin-shell.io/aas/3/0/AdministrativeInformation/"))
    graph.bind("aas-submodelelementcollection", Namespace("https://admin-shell.io/aas/3/0/SubmodelElementCollection/"))
    graph.bind("aas-qualifierkind", Namespace("https://admin-shell.io/aas/3/0/QualifierKind/"))
    graph.bind("aas-environment", Namespace("https://admin-shell.io/aas/3/0/Environment/"))
    graph.bind("aas-shortcuts", Namespace("https://admin-shell.io/aas/3/0/Shortcuts/"))
    graph.bind("aas", Namespace("https://admin-shell.io/aas/3/0/"))
    graph.bind("aas-multilanguageproperty", Namespace("https://admin-shell.io/aas/3/0/MultiLanguageProperty/"))


# Define the logic to execute when the button is clicked
#def execute_logic(button):
def aas_json_text_to_ttl(json_text: str) -> str:
    AAS = Namespace("https://admin-shell.io/aas/3/0/")
    #input_string = data_area.value      # Todo: hier raw data nutzen
    # Identify the input format (JSON/RDF)
    try:
        input_as_json: dict[str, Any] = json.loads(json_text)
    except JSONDecodeError as e:
        # Hier willst du wirklich JSON haben – deshalb hart fehlschlagen:
        raise ValueError("Eingabe ist kein gültiges JSON") from e
    #input_as_json = json.loads(input_string)
    # validate

    # Identify the model type
    if input_as_json.get('modelType') == "Submodel":
        graph, node = Submodel(**input_as_json).to_rdf()
    elif input_as_json.get('modelType') == "ConceptDescription":
        graph, node = ConceptDescription(**input_as_json).to_rdf()
    elif input_as_json.get('modelType') == "AssetAdministrationShell":
        graph, node = AssetAdministrationShell(**input_as_json).to_rdf()
    else:
        print("Processing Environment takes some time, wait!!!")
        graph = Graph()
        asset_administration_shells = input_as_json.get('assetAdministrationShells',[])
        print("processing asset administration shells")
        for asset_administration_shell in asset_administration_shells:
            try:
                sub_graph, _ = AssetAdministrationShell(**asset_administration_shell).to_rdf()
                graph = graph + sub_graph
            except:
                print("skipping AAS with id:"+asset_administration_shell['id'])
        print("processing submodels")
        submodels = input_as_json.get('submodels',[])
        for submodel in submodels:
            try:
                sub_graph, _ = Submodel(**submodel).to_rdf()
                graph = graph + sub_graph
            except:
                print("skipping AAS with id:"+submodel['id'])
        print("processing concept descriptions")
        concept_descriptions = input_as_json.get('conceptDescriptions',[])
        for concept_description in concept_descriptions:
            try:
                sub_graph, _ = ConceptDescription(**concept_description).to_rdf()
                graph = graph + sub_graph
            except:
                print("skipping AAS with id:"+concept_description['id'])


    # Bind the custom namespace with a prefix, otherwise, the output will look weird
    # This is one of the problems of AAS/RDF !
    bind_namespaces(graph)

    #output_area.value = graph.serialize(format="turtle_custom",base="https://company.com/aas/")
    # 'turtle_custom' ist das Format aus py-aas-rdf / Colab
    ttl_text: str = graph.serialize(format="turtle_custom",
                                    base="https://company.com/aas/")  # type: ignore[arg-type] # ToDo: Warum base so??!!
    return ttl_text

def json_file_to_ttl_file(json_path: Path, ttl_path: Path) -> None:
    """
    Nimmt eine JSON-Datei, wandelt sie nach RDF um und speichert sie als .ttl.
    """
    json_text = json_path.read_text(encoding="utf-8")
    ttl_text = aas_json_text_to_ttl(json_text)
    ttl_path.write_text(ttl_text, encoding="utf-8")