all: clean unzip unzip-old

unzip: dir 
	@for FILE in $(shell find . -maxdepth 1 -type f -name "*.tar.xz"); do tar xvJf $${FILE} -C unzipped/; done

unzip-old: dir 
	@for FILE in $(shell find old/ -maxdepth 1 -type f -name "*.tar.xz"); do tar xvJf $${FILE} -C unzipped/; done

zip: 
	@for FILE in $(shell find . -maxdepth 1 -type f -name "*.csv"); do tar cvJf $${FILE}.tar.xz $${FILE}; rm -f $${FILE}; done
	@for FILE in $(shell find old/ -maxdepth 1 -type f -name "*.csv"); do tar cvJf $${FILE}.tar.xz $${FILE}; rm -f $${FILE}; done

dir: 
	@mkdir -p unzipped/old/

clean: 
	@rm -rf unzipped/ 
