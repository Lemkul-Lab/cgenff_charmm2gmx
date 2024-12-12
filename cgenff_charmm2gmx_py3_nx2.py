#!/usr/bin/env python3

# USAGE: python cgenff_charmm2gmx.py DRUG drug.mol2 drug.str charmm36.ff
# Tested with Python 3.5.2 and Python 3.8. Requires numpy and networkx
# The networkx version MUST be in the 3.x series. Tested version: 3.3

# Copyright (C) 2014 E. Prabhu Raman prabhu@outerbanks.umaryland.edu
#
# Modified 11/6/2018 by Justin Lemkul to add lone pair support
# needed for CGenFF >= 4.0 halogens
#
# Modified 01/10/2019 by Conrard Tetsassi to work with Networkx 3.3
# Included notes on bonds, angles, and dihedrals
#
# For help/questions/bug reports, please contact Justin Lemkul jalemkul@vt.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU Affero General Public License for more details.
# <http://www.gnu.org/licenses/>

# EXAMPLE: You have a drug-like molecule in drug.mol2 file
# ParamChem returns a CHARMM stream file drug.str with topology and parameters
# INPUT
# The program needs four inputs:
#	(1) The first argument (resname) is found in the RESI entry in the CHARMM stream file; for example
#		RESI DRUG		  0.000  ! here DRUG is the resname
#	(2) drug.mol2 is the .mol2 which you supplied to ParamChem
#	(3) drug.str contains the topology entry and CGenFF parameters that ParamChem generates for you
#	(4) charmm36.ff should contain the CHARMM force-field in GROMACS format
#		Download it from: http://mackerell.umaryland.edu/CHARMM_ff_params.html

# OUTPUT
# The program will generate 4 output files ("DRUG" is converted to lowercase and the files are named accordingly):
#	(1) drug.itp - contains GROMACS itp
#	(2) drug.prm - contains parameters obtained from drug.str which are converted to GROMACS format and units
#	(3) drug.top - A Gromacs topology file which incorporates (1) and (2)
#	(4) drug_ini.pdb - Coordinates of the molecule obtained from drug.mol2

# The program has been tested only on CHARMM stream files containing topology and parameters of a single molecule.

import string
import re
import sys
import os
import math
import numpy as np
import networkx as nx

#=================================================================================================================
def check_versions(str_filename, ffdoc_filename):
    ffver = 0  # CGenFF version in force field directory
    strver = 0  # CGenFF version in stream file
    with open(str_filename, 'r') as f:
        for line in f:
            if line.startswith("* For use with CGenFF version"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) > 6:
                    strver = entry[6]
                    print("--Version of CGenFF detected in ", str_filename, ":", strver)
    with open(ffdoc_filename, 'r') as f:
        for line in f:
            if line.startswith("Parameters taken from CHARMM36 and CGenFF"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) > 6:
                    ffver = entry[6]
                    print("--Version of CGenFF detected in ", ffdoc_filename, ":", ffver)

    # warn the user about version mismatch
    if strver != ffver:
        print("\nWARNING: CGenFF versions are not equivalent!\n")

    # in case something has gone horribly wrong
    if (strver == 0) or (ffver == 0):
        print("\nERROR: Could not detect CGenFF version. Exiting.\n")
        exit()

#-----------------------------------------------------------------------
## jal
def is_lp(s):
    if len(s) >= 2 and s[0] == 'L' and s[1] == 'P':
        return True
    return False

#-----------------------------------------------------------------------
## jal
def is_lp_host_atom(self, name):
    for ai in range(self.nvsites):
        if name == self.G.nodes[ai]['at1']:
            return True
    return False

#-----------------------------------------------------------------------
## jal - only for COLINEAR lone pairs, since CGenFF only needs this now
def construct_lp(x1, y1, z1, x2, y2, z2, dist):
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    dr = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dr == 0:
        print("Error: Zero distance between constructing atoms for lone pair.")
        exit()
    dr = dist / dr
    # set LP coords
    xlp = x1 + dr * dx
    ylp = y1 + dr * dy
    zlp = z1 + dr * dz

    return xlp, ylp, zlp

#-----------------------------------------------------------------------
## jal
def find_vsite(self, atnum):
    for i in range(self.nvsites):
        # if we find the LP host, find the LP atom index
        if self.G.nodes[i]['at1'] == self.G.nodes[atnum]['name']:
            for j in range(self.natoms):
                if self.G.nodes[j]['vsite'] == self.G.nodes[i]['vsite']:
                    return j
    return None

#-----------------------------------------------------------------------
def read_gmx_atomtypes(filename):
    atomtypes = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith(";"):
                continue
            if line.strip() == '':
                continue
            entry = re.split(r'\s+', line.lstrip())
            if len(entry) >= 2:
                var = [entry[0], entry[1]]
                atomtypes.append(var)
    return atomtypes

#-----------------------------------------------------------------------
def get_filelist_from_gmx_forcefielditp(ffdir, ffparentfile):
    filelist = []
    with open(os.path.join(ffdir, ffparentfile), 'r') as f:
        for line in f:
            if line.startswith("#include"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) > 1:
                    filename = os.path.join(ffdir, entry[1].replace("\"", ""))
                    filelist.append(filename)
    return filelist

#-----------------------------------------------------------------------
def read_gmx_anglpars(filename):
    angllines = []
    with open(filename, 'r') as f:
        section = "NONE"
        for line in f:
            if line.startswith(";"):
                continue
            if line.strip() == '':
                continue
            if line.startswith("["):
                section = "NONE"
            if section == "ANGL":
                angllines.append(line)
            if line.startswith("[ angletypes ]"):
                section = "ANGL"

    anglpars = []
    for line in angllines:
        entry = re.split(r'\s+', line.lstrip())
        if len(entry) >= 5:
            ai, aj, ak, eq = entry[0], entry[1], entry[2], float(entry[4])
            anglpars.append([ai, aj, ak, eq])

    return anglpars

#-----------------------------------------------------------------------
def get_charmm_rtp_lines(filename, molname):
    foundmol = False
    store = False
    rtplines = []
    with open(filename, 'r') as f:
        for line in f:
            if store and line.startswith("RESI"):
                store = False

            if line.startswith("RESI"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) > 1 and entry[1] == molname:
                    store = True

            if line.startswith("END"):
                store = False

            if store:
                rtplines.append(line)

    return rtplines

#-----------------------------------------------------------------------
def get_charmm_prm_lines(filename):
    foundmol = False
    store = False
    prmlines = []
    with open(filename, 'r') as f:
        section = "NONE"
        for line in f:
            if line.startswith("END"):
                section = "NONE"
                store = False

            if store:
                prmlines.append(line)

            if line.startswith("read para"):
                section = "PRM"
                store = True

    return prmlines

#-----------------------------------------------------------------------
def parse_charmm_topology(rtplines):
    topology = {}
    noblanks = [x for x in rtplines if len(x.strip()) > 0]
    nocomments = [x for x in noblanks if x.strip()[0] not in ['*', '!']]
    section = "BONDS"  # default
    state = "free"
    for line in nocomments:
        if state == "free":
            if line.find("MASS") == 0:
                if "ATOMS" not in topology:
                    topology["ATOMS"] = {}
                s = line.split()
                if len(s) < 5:
                    continue
                idx, name, mass, type_ = int(s[1]), s[2], float(s[3]), s[4]
                comment = line[line.find("!")+1:].strip() if '!' in line else ""
                topology["ATOMS"][name] = [idx, mass, type_, comment]
            elif line.find("DECL") == 0:
                if "DECL" not in topology:
                    topology["DECL"] = []
                decl = line.split()[1]
                topology["DECL"].append(decl)
            elif line.find("DEFA") == 0:
                topology["DEFA"] = line[4:].strip()
            elif line.find("AUTO") == 0:
                topology["AUTO"] = line[4:].strip()
            elif line.find("RESI") == 0:
                if "RESI" not in topology:
                    topology["RESI"] = {}
                state = "resi"
                s = line.split()
                if len(s) < 3:
                    continue
                resname, charge = s[1], float(s[2])
                topology["RESI"][resname] = {
                    "charge": charge,
                    "cmaps": [],
                    "vsites": [],
                    "bonds": [],
                    "impropers": [],
                    "double_bonds": []
                }
                group = -1
            elif line.find("PRES") == 0:
                state = "pres"
                s = line.split()
                if len(s) < 3:
                    continue
                presname, charge = s[1], float(s[2])
            elif line.find("END") == 0:
                return topology
        elif state == "resi":
            if line.find("RESI") == 0:
                state = "resi"
                s = line.split()
                if len(s) < 3:
                    continue
                resname, charge = s[1], float(s[2])
                topology["RESI"][resname] = {
                    "charge": charge,
                    "cmaps": [],
                    "vsites": [],
                    "bonds": [],
                    "impropers": [],
                    "double_bonds": []
                }
                group = -1

            elif line.find("GROU") == 0:
                group += 1
                topology["RESI"][resname][group] = []
            elif line.find("ATOM") == 0:
                if '!' in line:
                    line = line[:line.find('!')]
                s = line.split()
                if len(s) < 4:
                    continue
                name, type_, charge = s[1], s[2], float(s[3])
                topology["RESI"][resname][group].append((name, type_, charge))
            ## jal - adding lone pair support
            elif line.find("LONE") == 0:
                if '!' in line:
                    line = line[:line.find('!')]
                s = line.split()
                if len(s) < 7:
                    continue
                name, at1, at2, dist = s[2], s[3], s[4], float(s[6]) * 0.1
                topology["RESI"][resname]["vsites"].append((name, at1, at2, dist))
            elif line.find("BOND") == 0 or line.find("DOUB") == 0:
                if '!' in line:
                    line = line[:line.find('!')]
                s = re.split(r'\s+', line.lstrip())
                numbonds = int((len(s) - 1) / 2)
                for bondi in range(numbonds):
                    p, q = s[1 + 2*bondi], s[2 + 2*bondi]
                    ## jal - ignore "bonds" to lone pairs
                    if not (is_lp(p) or is_lp(q)):
                        topology["RESI"][resname]["bonds"].append((p, q))
            elif line.find("DOUB") == 0:
                if '!' in line:
                    line = line[:line.find('!')]
                s = line.split()
                ndouble = int((len(s) - 1) / 2)
                for i in range(ndouble):
                    p, q = s[1 + 2*i], s[2 + 2*i]
                    topology["RESI"][resname]["double_bonds"].append((p, q))
            elif line.find("IMP") == 0:
                s = line.split()
                numimpr = int((len(s) - 2) / 4)
                for impr in range(numimpr):
                    ai = s[1 + 4*impr]
                    aj = s[2 + 4*impr]
                    ak = s[3 + 4*impr]
                    al = s[4 + 4*impr]
                    var = [ai, aj, ak, al]
                    topology["RESI"][resname]["impropers"].append(var)
            elif line.find("CMAP") == 0:
                if '!' in line:
                    line = line[:line.find('!')]
                s = line.split()
                cmap = s[1:9]
                if len(s) >= 9:
                    N = int(s[8])
                    cmapdata = []
                    while len(cmapdata) < N**2:
                        next_line = next(f, '').strip()
                        if next_line == '' or next_line.startswith('!'):
                            continue
                        cmapdata += list(map(float, next_line.split()))
                    if "CMAP" not in topology["RESI"][resname]:
                        topology["RESI"][resname]["cmaps"] = []
                    topology["RESI"][resname]["cmaps"].append([cmap, cmapdata])
            elif line.find("DONOR") == 0 or line.find("ACCEPTOR") == 0 or line.find("IC") == 0:
                continue

    return topology

#-----------------------------------------------------------------------
def parse_charmm_parameters(prmlines):
    parameters = {}
    cmapkey = ()
    noblanks = [x for x in prmlines if len(x.strip()) > 0]
    nocomments = [x for x in noblanks if x.strip()[0] not in ['*', '!']]
    section = "ATOM"  # default
    for line in nocomments:
        # print(line)
        sectionkeys = ["BOND", "ANGL", "DIHE",
                      "IMPR", "CMAP", "NONB", "HBON", "NBFI"]
        key = line.split()[0] if line.split() else ""
        if len(key) >= 4 and key[:4] in sectionkeys:
            section = key[:4]
            continue

        if section not in parameters:
            parameters[section] = []

        if section == "BOND":
            if '!' in line:
                line = line[:line.find('!')]
            s = re.split(r'\s+', line.lstrip())
            if len(s) < 4:
                continue
            ai, aj, kij, rij = s[0], s[1], float(s[2]), float(s[3])
            parameters["BOND"].append((ai, aj, kij, rij))
        elif section == "ANGL":
            if '!' in line:
                line = line[:line.find('!')]
            s = re.split(r'\s+', line.lstrip())
            if len(s) < 4:
                continue
            ai, aj, ak = s[0], s[1], s[2]
            other = list(map(float, s[3:]))
            parameters["ANGL"].append([ai, aj, ak] + other)
        elif section == "DIHE":
            if '!' in line:
                line = line[:line.find('!')]
            s = re.split(r'\s+', line.lstrip())
            if len(s) < 7:
                continue
            ai, aj, ak, al, k, n, d = s[0], s[1], s[2], s[3], float(s[4]), int(s[5]), float(s[6])
            parameters["DIHE"].append([ai, aj, ak, al, k, n, d])
        elif section == "IMPR":
            if '!' in line:
                line = line[:line.find('!')]
            s = re.split(r'\s+', line.lstrip())
            if len(s) < 7:
                continue
            ai, aj, ak, al, k, d = s[0], s[1], s[2], s[3], float(s[4]), float(s[6])
            parameters["IMPR"].append([ai, aj, ak, al, k, d])
        elif section == "CMAP":
            if '!' in line:
                line = line[:line.find('!')]
            if not cmapkey:
                s = line.split()
                if len(s) < 9:
                    continue
                a, b, c, d, e, f, g, h, N = s[0:9]
                cmapkey = (a, b, c, d, e, f, g, h, int(N))
                cmaplist = []
            else:
                cmapdata = list(map(float, line.split()))
                cmaplist += cmapdata
                if len(cmaplist) >= cmapkey[8]**2:
                    parameters["CMAP"].append([cmapkey, cmaplist[:cmapkey[8]**2]])
                    cmapkey = ()
        elif section == "NONB":
            if ("cutnb" in line.lower() or "wmin" in line.lower()):
                continue
            if '!' in line:
                comment = line[line.find('!')+1:]
                prm = line[:line.find('!')].split()
            else:
                comment = ""
                prm = line.split()
            if len(prm) < 4:
                continue
            atname = prm[0]
            try:
                epsilon = -float(prm[2])
                half_rmin = float(prm[3])
                parameters["NONB"].append((atname, epsilon, half_rmin))
            except ValueError:
                continue

            if len(prm) > 6:
                try:
                    epsilon14 = -float(prm[5])
                    half_rmin14 = float(prm[6])
                    if "NONBONDED14" not in parameters:
                        parameters["NONBONDED14"] = []
                    parameters["NONBONDED14"].append((atname, epsilon14, half_rmin14))
                except ValueError:
                    continue

    return parameters

#-----------------------------------------------------------------------
def write_gmx_bon(parameters, header_comments, filename):
    kcal2kJ = 4.18400

    with open(filename, "w") as outp:
        outp.write(f"{header_comments}\n")
        outp.write("[ bondtypes ]\n")
        kbond_conversion = 2.0 * kcal2kJ / (0.1)**2  # [kcal/mol]/A**2 -> [kJ/mol]/nm**2
        # factor of 0.5 because charmm bonds are Eb(r)=Kb*(r-r0)**2
        rbond_conversion = 0.1  # A -> nm
        outp.write(";%7s %8s %5s %12s %12s\n" % ("i", "j", "func", "b0", "kb"))
        if "BOND" in parameters:
            for p in parameters["BOND"]:
                ai, aj, kij, rij = p
                rij *= rbond_conversion
                kij *= kbond_conversion
                outp.write("%8s %8s %5i %12.8f %12.2f\n" % (ai, aj, 1, rij, kij))

        kangle_conversion = 2.0 * kcal2kJ  # [kcal/mol]/rad**2 -> [kJ/mol]/rad**2
        # factor of 0.5 because charmm angles are Ea(r)=Ka*(a-a0)**2
        kub_conversion = 2.0 * kcal2kJ / (0.1)**2  # [kcal/mol]/A**2 -> [kJ/mol]/nm**2prm
        ub0_conversion = 0.1  # A -> nm

        outp.write("\n\n[ angletypes ]\n")
        outp.write(";%7s %8s %8s %5s %12s %12s %12s %12s\n" %
                   ("i", "j", "k", "func", "theta0", "ktheta", "ub0", "kub"))
        if "ANGL" in parameters:
            for p in parameters["ANGL"]:
                if len(p) == 4:
                    ai, aj, ak, theta = p
                    kub = 0.0
                    ub0 = 0.0
                    kij = 0.0
                elif len(p) == 7:
                    ai, aj, ak, kij, theta, kub, ub0 = p
                else:
                    continue
                kij *= kangle_conversion
                kub *= kub_conversion
                ub0 *= ub0_conversion
                outp.write("%8s %8s %8s %5i %12.6f %12.6f %12.8f %12.2f\n" %
                           (ai, aj, ak, 5, theta, kij, ub0, kub))

        kdihe_conversion = kcal2kJ
        outp.write("\n\n[ dihedraltypes ]\n")
        outp.write(";%7s %8s %8s %8s %5s %12s %12s %5s\n" %
                   ("i", "j", "k", "l", "func", "phi0", "kphi", "mult"))
        if "DIHE" in parameters:
            for p in parameters["DIHE"]:
                if len(p) < 7:
                    continue
                ai, aj, ak, al, k, n, d = p
                k *= kdihe_conversion
                outp.write("%8s %8s %8s %8s %5i %12.6f %12.6f %5i\n" %
                           (ai, aj, ak, al, 9, d, k, n))

        kimpr_conversion = kcal2kJ * 2  # see above
        outp.write("\n\n[ dihedraltypes ]\n")
        outp.write("; 'improper' dihedrals \n")
        outp.write(";%7s %8s %8s %8s %5s %12s %12s\n" %
                   ("i", "j", "k", "l", "func", "phi0", "kphi"))
        if "IMPR" in parameters:
            for p in parameters["IMPR"]:
                if len(p) < 6:
                    continue
                ai, aj, ak, al, k, d = p
                k *= kimpr_conversion
                outp.write("%8s %8s %8s %8s %5i %12.6f %12.6f\n" %
                           (ai, aj, ak, al, 2, d, k))

    return

#-----------------------------------------------------------------------
def write_gmx_mol_top(filename, ffdir, prmfile, itpfile, molname):
    with open(filename, "w") as outp:
        outp.write("#include \"%s/forcefield.itp\"\n" % ffdir)
        outp.write("\n")
        outp.write("; additional params for the molecule\n")
        outp.write("#include \"%s\"\n" % prmfile)
        outp.write("\n")
        outp.write("#include \"%s\"\n" % itpfile)
        outp.write("\n")
        outp.write("#include \"%s/tip3p.itp\"\n" % ffdir)
        outp.write("#ifdef POSRES_WATER\n")
        outp.write("; Position restraint for each water oxygen\n")
        outp.write("[ position_restraints ]\n")
        outp.write(";  i funct		 fcx		fcy		   fcz\n")
        outp.write("   1	1		1000	   1000		  1000\n")
        outp.write("#endif\n")
        outp.write("\n")
        outp.write("; Include topology for ions\n")
        outp.write("#include \"%s/ions.itp\"\n" % ffdir)
        outp.write("\n")
        outp.write("[ system ]\n")
        outp.write("; Name\n")
        outp.write("mol\n")
        outp.write("\n")
        outp.write("[ molecules ]\n")
        outp.write("; Compound		  #mols\n")
        outp.write("%s			1\n" % molname)
        outp.write("\n")

#=================================================================================================================
class atomgroup:
    """
    A class that contains the data structures and functions to store and process
    data related to groups of atoms (read molecules)

    USAGE: m = atomgroup()
    """

    def __init__(self):
        self.G = nx.Graph()
        self.name = ""
        self.natoms = 0
        self.nvsites = 0
        self.nbonds = 0
        self.angles = []
        self.nangles = 0
        self.dihedrals = []
        self.ndihedrals = 0
        self.impropers = []
        self.nimpropers = 0
        # self.coord=np.zeros((self.natoms,3),dtype=float)

    #-----------------------------------------------------------------------
    def read_charmm_rtp(self, rtplines, atomtypes):
        """
        Reads CHARMM rtp
        Reads atoms, bonds, impropers
        Stores connectivity as a graph
        Autogenerates angles and dihedrals

        USAGE: m = atomgroup() ; m.read_charmm_rtp(rtplines, atomtypes)

        """
        # initialize everything
        self.G = nx.Graph()
        self.name = ""
        self.natoms = 0
        self.nvsites = 0
        self.nbonds = 0
        self.angles = []
        self.nangles = 0
        self.dihedrals = []
        self.ndihedrals = 0
        self.impropers = []
        self.nimpropers = 0

        atm = {}

        for line in rtplines:
            if '!' in line:
                line = line[:line.find('!')]

            if line.startswith("RESI"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) > 1:
                    self.name = entry[1]

            if line.startswith("ATOM"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) < 4:
                    continue
                atm[self.natoms] = {
                    'type': entry[2],
                    'resname': self.name,
                    'name': entry[1],
                    'charge': float(entry[3]),
                    'mass': float(0.00),
                    'beta': float(0.0),
                    'x': float(9999.9999),
                    'y': float(9999.9999),
                    'z': float(9999.9999),
                    'segid': self.name,
                    'resid': '1'
                }

                for typei in atomtypes:
                    if typei[0] == atm[self.natoms]['type']:
                        atm[self.natoms]['mass'] = float(typei[1])
                        break

                # Add node with attributes
                self.G.add_node(self.natoms, **atm[self.natoms])
                self.natoms += 1

            ## jal - adding lone pair support
            if line.startswith("LONE"):
                # entry = re.split('\s+', line.rstrip(line.lstrip()))
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) < 7:
                    continue
                atm[self.nvsites] = {
                    'vsite': entry[2],
                    'at1': entry[3],
                    'at2': entry[4],
                    'dist': float(entry[6]) * 0.1,
                    'x': float(9999.9999),
                    'y': float(9999.9999),
                    'z': float(9999.9999)
                }
                # DEBUG
                # print("Found lone pair in RTF: %s %s %s %.3f\n" % (atm[self.nvsites]['vsite'], atm[self.nvsites]['at1'], atm[self.nvsites]['at2'], atm[self.nvsites]['dist']))

                # Add lone pair as a node
                self.G.add_node(self.nvsites, **atm[self.nvsites])
                self.nvsites += 1

            if line.startswith("BOND") or line.startswith("DOUB"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) < 3:
                    continue
                numbonds = int((len(entry) - 1) / 2)
                for bondi in range(numbonds):
                    p, q = entry[1 + 2*bondi], entry[2 + 2*bondi]
                    # Find atom indices
                    found1 = False
                    found2 = False
                    for i in range(self.natoms):
                        if atm[i]['name'] == p:
                            idx1 = i
                            found1 = True
                            break
                    for j in range(self.natoms):
                        if atm[j]['name'] == q:
                            idx2 = j
                            found2 = True
                            break
                    if not found1:
                        print("Error:atomgroup:read_charmm_rtp> Atomname not found in top", p)
                        exit()
                    if not found2:
                        print("Error:atomgroup:read_charmm_rtp> Atomname not found in top", q)
                        exit()
                    ## jal - ignore "bonds" to lone pairs
                    if not (is_lp(atm[idx1]['name']) or is_lp(atm[idx2]['name'])):
                        self.G.add_edge(idx1, idx2, order='1')  # treat all bonds as single for now
                        self.nbonds += 1

            if line.startswith("IMP"):
                entry = re.split(r'\s+', line.lstrip())
                if len(entry) < 5:
                    continue
                numimpr = int((len(entry) - 2) / 4)
                for impri in range(numimpr):
                    if len(entry) < 2 + 4*impri + 4:
                        continue
                    ai = entry[1 + 4*impri]
                    aj = entry[2 + 4*impri]
                    ak = entry[3 + 4*impri]
                    al = entry[4 + 4*impri]
                    var = [ai, aj, ak, al]
                    self.impropers.append(var)

        self.nimpropers = len(self.impropers)
        if self.ndihedrals > 0 or self.nangles > 0:
            print("WARNING:atomgroup:read_charmm_rtp> Autogenerating angl-dihe even though they are preexisting", self.nangles, self.ndihedrals)
        self.autogen_angl_dihe()
        self.coord = np.zeros((self.natoms + self.nvsites, 3), dtype=float)

    #-----------------------------------------------------------------------
    def autogen_angl_dihe(self):
        self.angles = []
        for atomi in range(self.natoms):
            nblist = list(self.G.neighbors(atomi))
            for i in range(len(nblist) - 1):
                for j in range(i + 1, len(nblist)):
                    var = [nblist[i], atomi, nblist[j]]
                    self.angles.append(var)
        self.nangles = len(self.angles)
        self.dihedrals = []
        for i, j in self.G.edges():
            nblist1 = [nb for nb in self.G.neighbors(i) if nb != j]
            nblist2 = [nb for nb in self.G.neighbors(j) if nb != i]
            if len(nblist1) > 0 and len(nblist2) > 0:
                for ii in nblist1:
                    for jj in nblist2:
                        if ii != jj:
                            var = [ii, i, j, jj]
                            self.dihedrals.append(var)
        self.ndihedrals = len(self.dihedrals)

    #-----------------------------------------------------------------------
    def get_nonplanar_dihedrals(self, angl_params):
        nonplanar_dihedrals = []
        cutoff = 179.9
        for var in self.dihedrals:
            d1 = self.G.nodes[var[0]]['type']
            d2 = self.G.nodes[var[1]]['type']
            d3 = self.G.nodes[var[2]]['type']
            d4 = self.G.nodes[var[3]]['type']
            keep = 1
            for angl_param in angl_params:
                p1, p2, p3, eq = angl_param
                if d2 == p2 and ((d1 == p1 and d3 == p3) or (d1 == p3 and d3 == p1)):
                    if eq > cutoff:
                        keep = -1
                        break
                if d3 == p2 and ((d2 == p1 and d4 == p3) or (d2 == p3 and d4 == p1)):
                    if eq > cutoff:
                        keep = -1
                        break

            if keep == 1:
                nonplanar_dihedrals.append(var)

        return nonplanar_dihedrals

    #-----------------------------------------------------------------------
    def write_gmx_itp(self, filename, angl_params):
        with open(filename, 'w') as f:
            f.write("; Created by cgenff_charmm2gmx.py\n\n")
            f.write("[ moleculetype ]\n")
            f.write("; Name			   nrexcl\n")
            f.write(f"{self.name}				 3\n\n")
            f.write("[ atoms ]\n")
            f.write(";	 nr		  type	resnr residue  atom   cgnr	   charge		mass  typeB    chargeB		massB\n")
            f.write(f"; residue	 1 {self.name} rtp {self.name} q	qsum\n")
            pairs14 = nx.Graph()
            for atomi in range(self.natoms):
                pairs14.add_node(atomi)
                f.write("%6d %10s %6s %6s %6s %6d %10.3f %10.3f   ;\n" %
                        (atomi + 1, self.G.nodes[atomi]['type'],
                         self.G.nodes[atomi]['resid'], self.name, self.G.nodes[atomi]['name'], atomi + 1,
                         self.G.nodes[atomi]['charge'], self.G.nodes[atomi]['mass']))
            f.write("\n")
            f.write("[ bonds ]\n")
            f.write(";	ai	  aj funct			  c0			c1			  c2			c3\n")
            for i, j in self.G.edges():
                f.write("%5d %5d	 1 ;  %10s %10s\n" %
                        (i + 1, j + 1, self.G.nodes[i]['type'], self.G.nodes[j]['type']))
            f.write("\n")
            f.write("[ pairs ]\n")
            f.write(";	ai	  aj funct			  c0			c1			  c2			c3\n")
            for var in self.dihedrals:
                try:
                    path_length = len(nx.shortest_path(self.G, var[0], var[3]))
                except nx.NetworkXNoPath:
                    path_length = float('inf')
                if path_length == 4:  # this is to remove 1-2 and 1-3 included in dihedrals of rings
                    pairs14.add_edge(var[0], var[3])
            for i, j in pairs14.edges():
                f.write("%5d %5d	 1\n" % (i + 1, j + 1))
                # jal - add LP pairs, same as parent atom
                # Use is_lp_host_atom() to test each index, then find associated vsite
                if is_lp_host_atom(self, self.G.nodes[i]['name']):
                    k = find_vsite(self, i)
                    if k is not None:
                        f.write("%5d %5d	 1\n" % (k + 1, j + 1))
                if is_lp_host_atom(self, self.G.nodes[j]['name']):
                    k = find_vsite(self, j)
                    if k is not None:
                        f.write("%5d %5d	 1\n" % (k + 1, i + 1))
            f.write("\n")
            f.write("[ angles ]\n")
            f.write(";	ai	  aj	ak funct			c0			  c1			c2			  c3\n")
            for var in self.angles:
                f.write("%5d %5d %5d	5 ; %10s %10s %10s\n" % (var[0] + 1, var[1] + 1, var[2] + 1,
                                                                  self.G.nodes[var[0]]['type'], self.G.nodes[var[1]]['type'], self.G.nodes[var[2]]['type']))
            f.write("\n")
            f.write("[ dihedrals ]\n")
            f.write(";	ai	  aj	ak	  al funct			  c0			c1			  c2			c3			  c4			c5\n")
            nonplanar_dihedrals = self.get_nonplanar_dihedrals(angl_params)
            for var in nonplanar_dihedrals:
                f.write("%5d %5d %5d %5d	 9 ; %10s %10s %10s %10s\n" %
                        (var[0] + 1, var[1] + 1, var[2] + 1, var[3] + 1,
                         self.G.nodes[var[0]]['type'], self.G.nodes[var[1]]['type'],
                         self.G.nodes[var[2]]['type'], self.G.nodes[var[3]]['type']))
            f.write("\n")
            if self.nimpropers > 0:
                f.write("[ dihedrals ]\n")
                f.write(";	ai	  aj	ak	  al funct			  c0			c1			  c2			c3\n")
                for var in self.impropers:
                    # Assuming improper dihedrals use func=2
                    # You may need to adjust c0, c1, etc., based on your parameterization
                    f.write("%5d %5d %5d %5d	 2\n" % (var[0] + 1, var[1] + 1, var[2] + 1, var[3] + 1))
                f.write("\n")
            ## jal - add vsite directive
            ## we use 2fd construction, introduced in GROMACS-2020
            if self.nvsites > 0:
                func = 2
                f.write("[ virtual_sites2 ]\n")
                f.write("; Site   from				funct a\n")
                for atomi in range(self.nvsites):
                    vsite = 0
                    at1 = 0
                    at2 = 0
                    # find atom name matches
                    for ai in range(self.natoms):
                        if self.G.nodes[ai]['name'] == self.G.nodes[atomi]['vsite']:
                            vsite = ai
                        if self.G.nodes[ai]['name'] == self.G.nodes[atomi]['at1']:
                            at1 = ai
                        if self.G.nodes[ai]['name'] == self.G.nodes[atomi]['at2']:
                            at2 = ai
                    dist = self.G.nodes[atomi]['dist'] * -1  # invert sign for GROMACS convention
                    f.write("%5d %5d %5d %5d %8.3f\n" % (vsite + 1, at1 + 1, at2 + 1, func, dist))
                f.write("\n")

            ## jal - add exclusions for vsite
            if self.nvsites > 0:
                f.write("[ exclusions ]\n")
                f.write(";	ai	  aj\n")
                ## jal - explicitly add all 1-2, 1-3, and 1-4 exclusions
                ## for the lone pair, which are the same as 1-2, 1-3, 1-4
                ## exclusions for the host (bonds, angles, pairs)
                # first, exclude any LP from its host
                for i in range(self.natoms):
                    if is_lp_host_atom(self, self.G.nodes[i]['name']):
                        # find the LP attached to this host, not necessarily consecutive
                        # in the topology
                        j = find_vsite(self, i)
                        if j is not None:
                            f.write("%5d %5d	 1\n" % (i + 1, j + 1))
                # first neighbors: 1-2
                for i, j in self.G.edges():
                    if is_lp_host_atom(self, self.G.nodes[i]['name']):
                        k = find_vsite(self, i)
                        if k is not None:
                            f.write("%5d %5d	 1\n" % (k + 1, j + 1))
                    if is_lp_host_atom(self, self.G.nodes[j]['name']):
                        k = find_vsite(self, j)
                        if k is not None:
                            f.write("%5d %5d	 1\n" % (k + 1, i + 1))
                # second neighbors: 1-3
                for var in self.angles:
                    # only need to consider ends of the angle, not middle atom
                    ai = var[0]
                    ak = var[2]
                    if is_lp_host_atom(self, self.G.nodes[ai]['name']):
                        l = find_vsite(self, ai)
                        if l is not None:
                            f.write("%5d %5d	 1\n" % (l + 1, ak + 1))
                    if is_lp_host_atom(self, self.G.nodes[ak]['name']):
                        l = find_vsite(self, ak)
                        if l is not None:
                            f.write("%5d %5d	 1\n" % (l + 1, ai + 1))
                # third neighbors: 1-4
                for i, j in pairs14.edges():
                    if is_lp_host_atom(self, self.G.nodes[i]['name']):
                        k = find_vsite(self, i)
                        if k is not None:
                            f.write("%5d %5d	 1\n" % (k + 1, j + 1))
                    if is_lp_host_atom(self, self.G.nodes[j]['name']):
                        k = find_vsite(self, j)
                        if k is not None:
                            f.write("%5d %5d	 1\n" % (k + 1, i + 1))
                f.write("\n")

        f.close()

    #-----------------------------------------------------------------------
    def read_mol2_coor_only(self, filename):
        check_natoms = 0
        check_nbonds = 0
        with open(filename, 'r') as f:
            atm = {}
            section = "NONE"
            for line in f:
                secflag = False
                if line.startswith("@"):
                    secflag = True
                    section = "NONE"

                if section == "NATO" and not secflag:
                    entry = re.split(r'\s+', line.lstrip())
                    if len(entry) < 2:
                        continue
                    check_natoms = int(entry[0])
                    check_nbonds = int(entry[1])
                    if check_natoms != self.natoms:
                        # jal - if there are lone pairs, these will not be in the mol2 file
                        if self.nvsites == 0:
                            print("Error in atomgroup.py: read_mol2_coor_only: no. of atoms in mol2 (%d) and top (%d) are unequal" % (check_natoms, self.natoms))
                            print("Usually this means the specified residue name does not match between str and mol2 files")
                            exit()
                        else:
                            print("")
                            print("NOTE 5: %d lone pairs found in topology that are not in the mol2 file. This is not a problem, just FYI!\n" % self.nvsites)
                    # jal - if we have correctly ignored bonds to LP then there is no need
                    # for any check here
                    if check_nbonds != self.nbonds:
                        print("Error in atomgroup.py: read_mol2_coor_only: no. of bonds in mol2 (%d) and top (%d) are unequal" % (check_nbonds, self.nbonds))
                        exit()

                    section = "NONE"

                if section == "MOLE" and not secflag:
                    self.name = line.strip()
                    section = "NATO"  # next line after @<TRIPOS>MOLECULE contains atom, bond numbers

                if section == "ATOM" and not secflag:
                    entry = re.split(r'\s+', line.lstrip())
                    # guard against blank lines
                    if len(entry) > 1:
                        # jal - if there are lone pairs, these are not in mol2
                        # and are not necessarily something we can just tack on at the
                        # end of the coordinate section. Here, check the atom to see if it is
                        # the first constructing atom, and if so, we put in a dummy LP entry.
                        atomi = int(entry[0]) - 1
                        if atomi >= self.natoms + self.nvsites:
                            continue
                        self.G.nodes[atomi]['x'] = float(entry[2])
                        self.G.nodes[atomi]['y'] = float(entry[3])
                        self.G.nodes[atomi]['z'] = float(entry[4])
                        self.coord[atomi][0] = float(entry[2])
                        self.coord[atomi][1] = float(entry[3])
                        self.coord[atomi][2] = float(entry[4])
                        ## jal - if we have an atom that is the host for a LP, insert
                        ## the LP into the list
                        if is_lp_host_atom(self, self.G.nodes[atomi]['name']):
                            atomj = find_vsite(self, atomi)
                            if atomj is not None:
                                # insert dummy entry for LP
                                self.G.nodes[atomj]['x'] = float(9999.99)
                                self.G.nodes[atomj]['y'] = float(9999.99)
                                self.G.nodes[atomj]['z'] = float(9999.99)
                                self.coord[atomj][0] = float(9999.99)
                                self.coord[atomj][1] = float(9999.99)
                                self.coord[atomj][2] = float(9999.99)

                if line.startswith("@<TRIPOS>MOLECULE"):
                    section = "MOLE"
                elif line.startswith("@<TRIPOS>ATOM"):
                    section = "ATOM"
                elif line.startswith("@<TRIPOS>BOND"):
                    section = "BOND"

    #-----------------------------------------------------------------------
    def write_pdb(self, f):
        for atomi in range(self.natoms):
            if len(self.G.nodes[atomi]['name']) > 4:
                print("error in atomgroup.write_pdb(): atom name > 4 characters")
                exit()
            ## jal - construct LP sites
            if is_lp(self.G.nodes[atomi]['name']):
                # DEBUG
                # print("Found LP in write_pdb: %s\n" % self.G.nodes[atomi]['name'])
                # find constructing atoms, get their coordinates and construction distance*10
                atn1 = "dum"
                atn2 = "dum"
                dist = 0
                # loop over vsites
                for ai in range(self.nvsites):
                    if self.G.nodes[ai]['vsite'] == self.G.nodes[atomi]['name']:
                        atn1 = self.G.nodes[ai]['at1']  # atom name
                        atn2 = self.G.nodes[ai]['at2']  # atom name
                        dist = self.G.nodes[ai]['dist'] * 10  # Angstrom for PDB, was saved as *0.1 for GMX

                # get atom indices
                at1 = at2 = None
                for ai in range(self.natoms):
                    if self.G.nodes[ai]['name'] == atn1:
                        at1 = ai
                    if self.G.nodes[ai]['name'] == atn2:
                        at2 = ai

                # in case of failure
                if at1 is None or at2 is None:
                    print("Failed to match LP-constructing atoms in write_pdb!\n")
                    exit()

                # DEBUG
                # print("Found LP in write_pdb: %d %s %s with dist: %.3f\n" % ((atomi+1,at1+1,at2+1,dist)))

                # at1, at2, and dist only exist in vsite structure!
                x1, y1, z1 = self.coord[at1]
                x2, y2, z2 = self.coord[at2]

                xlp, ylp, zlp = construct_lp(x1, y1, z1, x2, y2, z2, dist)
                self.coord[atomi][0] = xlp
                self.coord[atomi][1] = ylp
                self.coord[atomi][2] = zlp
            f.write("%-6s%5d %-4s %-4s%5s%12.3f%8.3f%8.3f%6.2f%6.2f\n" %
                    ("ATOM", atomi + 1, self.G.nodes[atomi]['name'], self.name, self.G.nodes[atomi]['resid'],
                     self.coord[atomi][0], self.coord[atomi][1], self.coord[atomi][2],
                     1.0, self.G.nodes[atomi]['beta']))
        f.write("END\n")

#=================================================================================================================


if len(sys.argv) != 5:
    print("Usage: RESNAME drug.mol2 drug.str charmm36.ff")
    exit()

# Check for compatible NetworkX version
if float(nx.__version__) < 2.0:
    print("Your NetworkX version is: ", nx.__version__)
    print("This script requires a version in the 2.x or 3.x series")
    print("Your NetworkX package is incompatible with this conversion script and cannot be used.")
    exit()
else:
    if float(nx.__version__) > 3.0:
        print("This script may not have been tested with NetworkX versions newer than 3.0.")
        print("Proceeding with caution.")

if sys.version_info < (3, 0):
    print("You are using a Python version in the 2.x series. This script requires Python 3.0 or higher.")
    print("Please visit http://mackerell.umaryland.edu/charmm_ff.shtml#gromacs to get a script for Python 3.x")
    exit()

mol_name = sys.argv[1]
mol2_name = sys.argv[2]
rtp_name = sys.argv[3]
ffdir = sys.argv[4]
atomtypes_filename = os.path.join(ffdir, "atomtypes.atp")

print("NOTE 1: Code tested with Python above 3.5. Your version:", sys.version)
print("")
print("NOTE 2: Code tested with NetworkX above 2.3. Your version:", nx.__version__)
print("")
print("NOTE 3: Please be sure to use the same version of CGenFF in your simulations that was used during parameter generation:")
check_versions(rtp_name, os.path.join(ffdir, "forcefield.doc"))
print("")
print("NOTE 4: To avoid duplicated parameters, do NOT select the 'Include parameters that are already in CGenFF' option when uploading a molecule into CGenFF.")

# For output
itpfile = mol_name.lower() + ".itp"
prmfile = mol_name.lower() + ".prm"
initpdbfile = mol_name.lower() + "_ini.pdb"
topfile = mol_name.lower() + ".top"

atomtypes = read_gmx_atomtypes(atomtypes_filename)

angl_params = []  # needed for detecting triple bonds
filelist = get_filelist_from_gmx_forcefielditp(ffdir, "forcefield.itp")
for filename in filelist:
    anglpars = read_gmx_anglpars(filename)
    angl_params += anglpars

m = atomgroup()
rtplines = get_charmm_rtp_lines(rtp_name, mol_name)
m.read_charmm_rtp(rtplines, atomtypes)

m.read_mol2_coor_only(mol2_name)
with open(initpdbfile, 'w') as f:
    m.write_pdb(f)

prmlines = get_charmm_prm_lines(rtp_name)
params = parse_charmm_parameters(prmlines)
write_gmx_bon(params, "", prmfile)
anglpars = read_gmx_anglpars(prmfile)
angl_params += anglpars  # append the new angl params

m.write_gmx_itp(itpfile, angl_params)
write_gmx_mol_top(topfile, ffdir, prmfile, itpfile, mol_name)

print("============ DONE ============")
print("Conversion complete.")
print("The molecule topology has been written to %s" % itpfile)
print("Additional parameters needed by the molecule are written to %s, which needs to be included in the system .top" % prmfile)
print("\nPLEASE NOTE: If your topology has lone pairs, you must use GROMACS version 2020 or newer to use 2fd construction")
print("Older GROMACS versions WILL NOT WORK as they do not support 2fd virtual site construction\n")
print("============ DONE ============")

exit()

