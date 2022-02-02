CoLA
The Corpus of Linguistic Acceptability
CoLA - The Corpus of Linguistic Acceptability
http://nyu-mll.github.io/cola

0. Authors

Alex Warstadt
Amanpreet Singh
Sam Bowman
(New York University)


1. Introduction

The Corpus of Linguistic Acceptability (CoLA) in its full form consists of 10657 sentences from 23 linguistics publications, expertly annotated for acceptability (grammaticality) by their original authors. The public version provided here contains 9594 sentences belonging to training and development sets, and excludes 1063 sentences belonging to a held out test set. Contact alexwarstadt [at] gmail [dot] com with any questions or issues.


2. Download

Download the dataset at: http://nyu-mll.github.io/cola


3. Citation

@misc{warstadt-18,
Author = {Warstadt, Alexander and Singh, Amanpreet 
and Bowman, Samuel R},
Howpublished = {http://nyu-mll.github.io/cola},
Title = {Corpus of Linguistic Acceptability},
Year = {2018}}


4. Data description

4.1 Split

We have split the data into an in-domain set comprised sentences from 17 sources and an out-of-domain set comprised of the remaining 6 sources. The in-domain set is split into train/dev/test sets, and the out-of-domain is split into dev/test sets. The test sets are not made public. For convenience, each dataset is provided is provided twice, in raw form, and tokenized form. We used the NLTK tokenizer. The public data is split into the following files:
raw/in_domain_train.tsv (8551 lines)
raw/in_domain_dev.tsv (527 lines)
raw/out_of_domain_dev.tsv (516 lines)
tokenized/in_domain_train.tsv (8551 lines)
tokenized/in_domain_dev.tsv (527 lines)
tokenized/out_of_domain_dev.tsv (516 lines)

4.2 Data Format

Each line in the .tsv files consists of 4 tab-separated columns.
Column 1:	the code representing the source of the sentence.
Column 2:	the acceptability judgment label (0=unacceptable, 1=acceptable).
Column 3:	the acceptability judgment as originally notated by the author.
Column 4:	the sentence.


4.3 Corpus Sample

clc95	0	*	In which way is Sandy very anxious to see if the students will be able to solve the homework problem?
c-05	1		The book was written by John.
c-05	0	*	Books were sent to each other by the students.
swb04	1		She voted for herself.
swb04	1		I saw that gas can explode.

4.4 Processing

During gathering of the data and processing, some sentences from the source documents may have been omitted or altered. We retained all acceptable examples, and excluded any examples given intermediate judgments such as “?” or “#”. In addition, we excluded examples of unacceptable sentences not suitable for the present task because they required reasoning about pragmatic violations, unavailable semantic readings, or nonexistent words. We take responsibility for any errors.


5. Disclaimer

The text in this corpus is excerpted from the published works at the end of this document, and copyright (where applicable) remains with the original authors or publishers. We expect that research use within the US is legal under fair use, but make no guarantee of this.

The sentences were gathered from 23 sources, with full citations given below. Those sources were divided into two categories: 17 in-domain, and 6 out-of-domain (development and test only). Each source is associated with a code in the dataset. Below find a list of each in-domain and out-of-domain sources, its identifier code, and its size.


6. Sources

In domain: 
(Source, Code, N)
Adger (2003), ad-03, 948
Baltin (1982), b_82, 96
Baltin and Collins (2001), bc01, 880
Bresnan (1973), b_73, 259
Carnie (2013), c_13, 870
Culicover and Jackendoff (1999), cj99, 233
Dayal (1998), d_98, 179
Gazdar (1981), g_81, 110
Goldberg and Jackendoff (2004), gj04, 106
Kadmon and Landman (1993), kl93, 93
Kim and Sells (2008), ks08, 1965
Levin (1993), l-93 1459
Miller (2002), m_02, 426
Rappaport Hovav and Levin (2008), rhl08, 151
Ross (1967), r-67, 1029
Sag et al. (1985), sgww85, 153
Sportiche et al. (2013), sks13, 651

Out of domain: 
(Source, Code, N)
Chung et al. (1995), clc95, 148
Collins (2005), c-05, 66
Jackendoff (1971), j_71, 94
Sag (1997), s_97, 112
Sag et al. (1999), swb04, 460
Williams (1980), w_80, 169


7. References

David Adger. 2003. Core syntax: A minimalist approach. Oxford University Press Oxford.
Mark Baltin and Chris Collins, editors. 2001. Handbook of Contemporary Syntactic Theory. Blackwell Publishing Ltd.
Mark R Baltin. 1982. A landing site theory of movement rules. Linguistic Inquiry, 13(1):1–38.
Joan W Bresnan. 1973. Syntax of the comparative clause construction in english. Linguistic inquiry, 4(3):275– 343.
Andrew Carnie. 2013. Syntax: A generative introduction. John Wiley & Sons.
Sandra Chung, William A Ladusaw, and James McCloskey. 1995. Sluicing and logical form. Natural language semantics, 3(3):239–282.
Chris Collins. 2005. A smuggling approach to the passive in english. Syntax, 8(2):81–120.
Peter W Culicover and Ray Jackendoff. 1999. The view from the periphery: The english comparative correlative. Linguistic inquiry, 30(4):543–571.
Veneeta Dayal. 1998. Any as inherently modal. Linguistics and philosophy, 21(5):433–476.
Gerald Gazdar. 1981. Unbounded dependencies and coordinate structure. In The Formal complexity of natural language, pages 183–226. Springer.
Adele E Goldberg and Ray Jackendoff. 2004. The English resultative as a family of constructions. Language, 80(3):532–568.
Ray S Jackendoff. 1971. Gapping and related rules. Linguistic inquiry, 2(1):21–35.
Nirit Kadmon and Fred Landman. 1993. Any. Linguistics and Philosophy, 16(4):353–422, aug.
Jong-Bok Kim and Peter Sells. 2008. English syntax: An introduction. CSLI publications.
Beth Levin. 1993. English verb classes and alternations: A preliminary investigation. University of Chicago press.
Jim Miller. 2002. An introduction to English syntax. Edinburgh Univ Press.
Malka Rappaport Hovav and Beth Levin. 2008. The english dative alternation: The case for verb sensitivity1. Journal of linguistics, 44(1):129–167.
John Robert Ross. 1967. Constraints on variables in syntax. Ph.D. thesis, MIT.
Ivan A Sag, Gerald Gazdar, Thomas Wasow, and Steven Weisler. 1985. Coordination and how to distinguish categories. Natural Language & Linguistic Theory, 3(2):117–171.
Ivan A Sag. 1997. English relative clause constructions. Journal of linguistics, 33(2):431–483.
Dominique Sportiche, Hilda Koopman, and Edward Stabler. 2013. An introduction to syntactic analysis and theory. John Wiley & Sons.
Edwin Williams. 1980. Predication. Linguistic inquiry, 11(1):203–238.