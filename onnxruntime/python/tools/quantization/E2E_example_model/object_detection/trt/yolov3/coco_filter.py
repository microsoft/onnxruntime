import json
import os
from pathlib import Path

class CocoFilter():
    """ Filters the COCO dataset
    """
    def _process_info(self):
        self.info = self.coco['info']
        
    def _process_licenses(self):
        self.licenses = self.coco['licenses']
        
    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()
        self.category_to_image_ids = dict()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                self.category_set.add(category['name'])
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')
            
            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id} # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):
        self.images = dict()
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
                
    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def _filter_categories(self):
        """ Find category ids matching args
            Create mapping from original category id to new category id
            Create new collection of categories
        """
        if 'all' in self.filter_categories:
            print("Filter all categories.")
            self.filter_categories = self.category_set

        missing_categories = set(self.filter_categories) - self.category_set
        if len(missing_categories) > 0:
            print(f'Did not find categories: {missing_categories}')
            should_continue = input('Continue? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()

        self.new_category_map = dict()
        new_id = 1
        for key, item in self.categories.items():
            if item['name'] in self.filter_categories:
                self.new_category_map[key] = new_id
                new_id += 1
            else:
                print(item['name'])

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category['id'] = int(new_id)
            self.new_categories.append(new_category)

    def _handle_images(self):
        for image_id, segmentation_list in self.segmentations.items():
            for segmentation in segmentation_list:
                original_seg_cat = segmentation['category_id']
                cat_id = original_seg_cat

                if cat_id not in self.new_category_map.keys():
                    continue

                if cat_id not in self.category_to_image_ids:
                    self.category_to_image_ids[cat_id] = set()
                self.category_to_image_ids[cat_id].add(image_id)

        for key, value in self.category_to_image_ids.items():
            self.category_to_image_ids[key] = sorted(list(value))
            import random
            random.shuffle(self.category_to_image_ids[key])

        self.calib_img_list = set() 

        for key, value in self.category_to_image_ids.items():
            for i in value[:20]:
                self.calib_img_list.add(i)


        import requests
        for id in self.calib_img_list:
            im = self.images[id]
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(self.image_folder, 'calib', im['file_name']), 'wb') as handler:
                handler.write(img_data)

    def main(self, args):
        # Open json
        self.input_json_path = Path(args.input_json)
        self.image_folder = Path(args.image_folder)
        self.filter_categories = args.categories

        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()

        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
        
        # Process the json
        print('Processing input json...')
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

        # Filter to specific categories
        print('Filtering...')
        self._filter_categories()
        self._handle_images()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
    "Filters a COCO Instances JSON file to only include specified categories. "
    "This includes images. Does not modify 'info' or 'licenses'.")
    
    parser.add_argument("-i", "--input_json", dest="input_json",
        help="path to a json file in coco format")
    parser.add_argument("-f", "--image_folder", dest="image_folder",
        help="folder to save images")
    parser.add_argument("-c", "--categories", nargs='+', dest="categories",
        help="List of category names separated by spaces, e.g. -c person dog bicycle. If -c all, it includes all categories.")

    args = parser.parse_args()

    cf = CocoFilter()
    cf.main(args)
