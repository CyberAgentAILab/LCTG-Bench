.PHONY: all
all: help

.PHONY: help ## View help

.PHONY: run ## Run docker image at local machine
run:
	bash run_lctg.sh

help:
	@grep -E '^.PHONY: [a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | sed 's/^.PHONY: //g' | awk 'BEGIN {FS = "## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'