/*
 * Copyright (c) 2013-2016 Los Alamos National Security, LLC
 *                         All rights reserved.
 *
 * This software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 * National Security, LLC for the U.S. Department of Energy. The U.S. Government
 * has rights to use, reproduce, and distribute this software.  NEITHER THE
 * GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
 * OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If
 * software is modified to produce derivative works, such modified software
 * should be clearly marked, so as not to confuse it with the version available
 * from LANL.
 *
 * Additionally, redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following conditions
 * are met:
 *
 * · Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * · Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * · Neither the name of Los Alamos National Security, LLC, Los Alamos
 *   National Laboratory, LANL, the U.S. Government, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL
 * SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file quo-set.c
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "quo-set.h"
#include "quo-private.h"
#include "quo.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif
#ifdef HAVE_STDBOOL_H
#include <stdbool.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#include <errno.h>

/* ////////////////////////////////////////////////////////////////////////// */
/**
 * This is a naive algorithm to compute the intersection of k sets. if this
 * approach ever becomes a performance bottleneck, then implement Baeza-Yates.
 *
 * Caller is responsible for freeing returned resources.
 */
int
quo_set_get_k_set_intersection(int nsets,
                               const int *set_lens,
                               int **sets,
                               int **res,
                               int *res_len)
{
    /* all set data are positive, so we don't have to worry about that */
    int global_max = -1;
    /* "hash table" large enough to hold the largest set value. */ 
    int *big_htab = NULL;
    /* length of the k set intersection */
    int ilen = 0;
    /* size of big_htab */
    size_t big_htab_size = 0;

    if (!set_lens || !sets || !res || !res_len) return QUO_ERR_INVLD_ARG;
    *res = NULL; *res_len = 0;
    if (0 == nsets) return QUO_SUCCESS; /* nothing to do */

    /* sanity - make sure that every array is sorted in increasing order and has
     * no duplicates. also find the max value across all sets. */
    for (int set = 0; set < nsets; ++set) {
        int curval = 0;
        for (int elem = 0; elem < set_lens[set]; ++elem) {
            if (0 == elem) {
                curval = sets[set][elem]; 
            }
            else {
                if (curval >= sets[set][elem]) return QUO_ERR_INVLD_ARG;
                curval = sets[set][elem];
            }
            /* make sure that we are dealing with positive values. */
            if (curval < 0) return QUO_ERR_INVLD_ARG;
            if (global_max < curval) global_max = curval;
        }
    }
    /* now that we know the global max, allocate an array large enough to
     * accommodate that range. remember, values here should be from
     * 0 - global_max. */
    big_htab_size = (global_max + 1) * sizeof(*big_htab);
    if (NULL == (big_htab = malloc(big_htab_size))) {
        QUO_OOR_COMPLAIN();
        return QUO_ERR_OOR;
    }
    /* -1 indicates that the hash location has not yet been taken */
    (void)memset(big_htab, -1, big_htab_size);
    /* ////////////////////////////////////////////////////////////////////// */
    /* start the naive k set intersection */
    /* ////////////////////////////////////////////////////////////////////// */
    for (int seta = 0; seta < nsets; ++seta) {
        for (int setb = 0; setb < nsets; ++setb) {
            int i = 0, j = 0;
            if (seta == setb) continue;
            while (i < set_lens[seta] && j < set_lens[setb]) {
                /* no possible intersection */
                if (0 == set_lens[seta] || 0 == set_lens[setb]) break;
                while (i < set_lens[seta] && sets[seta][i] < sets[setb][j]) ++i;
                if (i == set_lens[seta]) break;
                while (j < set_lens[setb] && sets[setb][j] < sets[seta][i]) ++j;
                if (j == set_lens[setb]) break;
                if (sets[seta][i] == sets[setb][j]) {
                    /* new entry */
                    if (-1 == big_htab[sets[seta][i]]) {
                        big_htab[sets[seta][i]] = sets[seta][i];
                        ++ilen;
                    }
                }
                ++i; ++j;
            }
        }
    }
    /* if no intersections found, we are done! */
    if (0 == ilen) goto done;
    /* else return the set */
    if (NULL == (*res = calloc(ilen, sizeof(int)))) {
        free(big_htab); big_htab = NULL;
        QUO_OOR_COMPLAIN();
        return QUO_ERR_OOR;
    }
    /* populate the result array - note this will always be sorted */
    for (int i = 0, j = 0; i < global_max + 1; ++i) {
        if (-1 != big_htab[i]) (*res)[j++] = big_htab[i];
    }
    /* return result array length */
    *res_len = ilen;
#if 0 /* debug */
    for (int i = 0; i < ilen; ++i) {
        printf("%d%s", (*res)[i], i + 1 == ilen ? "\n" : " ");
    }
#endif
done:
    if (big_htab) free(big_htab);
    return QUO_SUCCESS;
}
