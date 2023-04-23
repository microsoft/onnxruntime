diff \
  <(zcat optest.main.log.gz     | sed -E 's#^2022[^]]*\]##g' | sed -E 's#\([0-9]* ms( total)?\)##g') \
  <(zcat optest.refactor.log.gz | sed -E 's#^2022[^]]*\]##g' | sed -E 's#\([0-9]* ms( total)?\)##g')
