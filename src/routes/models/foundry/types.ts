// Type definitions for Azure AI Foundry models

export interface FoundryModel {
	id: string;
	name: string;
	version: string;
	description: string;
	longDescription?: string;
	deviceSupport: string[];
	tags: string[];
	publisher: string;
	lastModified: string;
	createdDate: string;
	downloadCount?: number;
	framework?: string;
	license?: string;
	taskType?: string;
	modelSize?: string;
	inputFormat?: string;
	outputFormat?: string;
	sampleCode?: string;
	documentation?: string;
	githubUrl?: string;
	paperUrl?: string;
	demoUrl?: string;
	benchmarks?: Benchmark[];
	requirements?: string[];
	compatibleVersions?: string[];
}

// New type for grouped models by alias
export interface GroupedFoundryModel {
	alias: string; // Short name like "deepseek-r1-8b"
	displayName: string; // Pretty name for display
	description: string;
	longDescription?: string;
	deviceSupport: string[]; // Combined devices from all variants
	tags: string[]; // Combined tags from all variants
	publisher: string;
	lastModified: string; // Latest modified date
	createdDate: string; // Earliest created date
	downloadCount?: number; // Sum of all downloads
	framework?: string; // Primary framework
	license?: string;
	taskType?: string;
	modelSize?: string;
	variants: FoundryModel[]; // All device variants
	// Additional computed fields
	availableDevices: string[]; // Devices this model supports
	totalDownloads: number;
	latestVersion: string;
	documentation?: string;
	githubUrl?: string;
	paperUrl?: string;
	demoUrl?: string;
	benchmarks?: Benchmark[];
	requirements?: string[];
	compatibleVersions?: string[];
}

export interface Benchmark {
	metric: string;
	value: string;
	device: string;
}

export interface FilterOptions {
	devices: string[];
	frameworks: string[];
	taskTypes: string[];
	publishers: string[];
}

export interface SortOption {
	key: string;
	label: string;
}

export const SORT_OPTIONS: SortOption[] = [
	{ key: 'displayName', label: 'Name' },
	{ key: 'lastModified', label: 'Last Modified' },
	{ key: 'totalDownloads', label: 'Downloads' },
	{ key: 'publisher', label: 'Publisher' },
	{ key: 'modelSize', label: 'Size' }
];

export const DEVICE_ICONS: Record<string, string> = {
	'npu': '🧠',
	'gpu': '🎮',
	'cpu': '💻',
	'fpga': '⚡',
	'asic': '🔧'
};

export const SUPPORTED_DEVICES = [
	'npu',
	'gpu', 
	'cpu',
	'fpga',
	'asic'
];

export const TASK_TYPE_COLORS: Record<string, string> = {
	'Text Generation': 'badge-primary',
	'Object Detection': 'badge-secondary',
	'Speech Recognition': 'badge-accent',
	'Image Classification': 'badge-info',
	'General': 'badge-neutral'
};