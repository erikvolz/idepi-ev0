#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

#
# mrmr.py :: computes Maximum Relevance (MaxRel) and minimum
# Redundancy Maximum Relevance (mRMR) for a dataset
#
# Copyright (C) 2011 N Lance Hepler <nlhepler@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

NUM_COLS = 2000

import os, sys, time

import multiprocessing as mp
import numpy as np

from idepi import DiscreteMrmr, FastCaim, UiThread, GaussianKde, MixedMrmr, PhyloMrmr


THRESHOLD = 0.8


def __usage():
    print >> sys.stderr, 'usage: %s MRMRFILE' % __name
    return -1


def find_files(prefix):
    if os.path.exists(prefix):
        return (prefix,)
    else:
        suffixes = ('.colnames.json', '.x.npz', '.y.npz')
        for suffix in suffixes:
            if not os.path.exists(prefix + suffix):
                return None
        return tuple([prefix + suffix for suffix in suffixes])


def main(argv, ui=None):

    normalized = False
    klass = None
    num_features = 10

    i = 0
    while i < len(argv):
        if argv[i][0] == '-':
            opt = argv[i][1]
            if opt is 'n':
                num_features = int(argv[i+1])
                i += 2
                continue
            elif opt is 'd':
                klass = DiscreteMrmr
                i += 1
                continue
            elif opt is 'c':
                klass = MixedMrmr
                i += 1
                continue
            elif opt is 'p':
                klass = PhyloMrmr
                i += 1
                continue
            elif opt is 'N':
                normalized = True
                i += 1
                continue
            elif opt is '-':
                if argv[i][2:] == 'gpl':
                    print 'Go look up the GNU GPLv2 on gnu.org'
                    return 0
            return usage()
        else:
            argv = argv[i:]
            break

    if klass is None:
        klass = DiscreteMrmr

    files = find_files(argv[0])

    if len(argv) != 1 or files is None:
        return usage()

    if len(files) == 1:
        fh = open(argv[0], 'r')
        lines = [l.strip() for l in fh]
        fh.close()

        names = lines[0].split(',')[1:]
        arr = [[float(x) for x in l.split(',')] for l in lines[1:] if l != '']
        m = np.array(arr, dtype=float)

        targets = m[:, 0].astype(bool)
        variables = m[:, 1:]
    elif len(files) == 3:
        for suffix in ('.colnames.json', '.x.npz', '.y.npz'):
            if not os.path.exists(argv[0] + suffix):
                usage()
        with open(argv[0] + '.colnames.json', 'r') as fh:
            import json
            names = json.load(fh)
        variables = np.load(argv[0] + '.x.npz')['arr_0'].tolist()['data']
        targets = np.load(argv[0] + '.y.npz')['arr_0'].tolist()['data'].astype(bool)
    else:
        usage()

    if klass != PhyloMrmr and len(files) == 3:
        variables = variables[:, :]['b']
    elif klass != PhyloMrmr:
        variables = variables.astype(int)

    nrow, ncol = variables.shape

    if klass == MixedMrmr:
        # variables = variables[:, :NUM_COLS]
        fc = FastCaim()
        print 'starting discretization'
        b = time.time()
        fc.learn(variables, targets)
        variables = fc.discretize(variables)
        print 'done discretizing %i columns, took' % variables.shape[1], time.time() - b, 'seconds'
        klass = DiscreteMrmr

    if normalized:
        klass._NORMALIZED = True
    else:
        klass._NORMALIZED = False

    selector = klass(num_features, klass.MID, THRESHOLD)

    print 'starting feature selection'
    b = time.time()

    # hax to get at 'private' method
    maxrel, mrmr = selector._mrmr_selection(num_features, klass.MID, variables, targets, threshold=THRESHOLD, ui=ui)

    print 'done feature selection across %i columns, took' % variables.shape[1], time.time() - b, 'seconds'

    print 'I(X, Y) / H(X, Y)'
    for idx, value in maxrel:
        print '   %4d   % 5s   %6.4f' % (idx + 1, names[idx], value)

    print '\nI(X, Y) / H(X, Y) - I(X, X) / H(X, X) (related > %.3g)' % THRESHOLD
    for idx, value, related in mrmr:
        print '   %4d   % 5s   %6.4f   (%s)' % (idx + 1, names[idx], value, '%d' % len(related)) #', '.join([names[i] + ': %6.4f' % v for i, v in related]))

    # selector.select(variables, targets)
    # print selector.subset(variables)

    return 0


if __name__ == '__main__':
    __name = os.path.basename(sys.argv[0])
    usage = __usage
    ui = UiThread()
    sys.exit(main(sys.argv[1:], ui))
